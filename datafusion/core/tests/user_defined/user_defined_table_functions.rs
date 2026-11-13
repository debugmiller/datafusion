// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::fs::File;
use std::io::Seek;
use std::path::Path;
use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::csv::ReaderBuilder;
use arrow::csv::reader::Format;

use arrow::datatypes::{DataType, Field};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::test_util::batches_to_string;
use datafusion::datasource::TableProvider;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::SessionContext;
use datafusion_catalog::TableFunctionImpl;
use datafusion_catalog::{Session, TableFunctionArgs};
use datafusion_common::{DFSchema, ScalarValue, assert_contains, plan_err};
use datafusion_expr::{
    EmptyRelation, Expr, LogicalPlan, Projection, Signature, TableType, Volatility,
};

use async_trait::async_trait;

/// test simple udtf with define read_csv with parameters
#[tokio::test]
async fn test_simple_read_csv_udtf() -> Result<()> {
    let ctx = SessionContext::new();

    ctx.register_udtf("read_csv", Arc::new(SimpleCsvTableFunc {}));

    let csv_file = "tests/tpch-csv/nation.csv";
    // read csv with at most 5 rows
    let rbs = ctx
        .sql(format!("SELECT * FROM read_csv('{csv_file}', 5);").as_str())
        .await?
        .collect()
        .await?;

    insta::assert_snapshot!(batches_to_string(&rbs), @r"
    +-------------+-----------+-------------+-------------------------------------------------------------------------------------------------------------+
    | n_nationkey | n_name    | n_regionkey | n_comment                                                                                                   |
    +-------------+-----------+-------------+-------------------------------------------------------------------------------------------------------------+
    | 1           | ARGENTINA | 1           | al foxes promise slyly according to the regular accounts. bold requests alon                                |
    | 2           | BRAZIL    | 1           | y alongside of the pending deposits. carefully special packages are about the ironic forges. slyly special  |
    | 3           | CANADA    | 1           | eas hang ironic, silent packages. slyly regular packages are furiously over the tithes. fluffily bold       |
    | 4           | EGYPT     | 4           | y above the carefully unusual theodolites. final dugouts are quickly across the furiously regular d         |
    | 5           | ETHIOPIA  | 0           | ven packages wake quickly. regu                                                                             |
    +-------------+-----------+-------------+-------------------------------------------------------------------------------------------------------------+
    ");

    // just run, return all rows
    let rbs = ctx
        .sql(format!("SELECT * FROM read_csv('{csv_file}');").as_str())
        .await?
        .collect()
        .await?;

    insta::assert_snapshot!(batches_to_string(&rbs), @r"
    +-------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------+
    | n_nationkey | n_name    | n_regionkey | n_comment                                                                                                          |
    +-------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------+
    | 1           | ARGENTINA | 1           | al foxes promise slyly according to the regular accounts. bold requests alon                                       |
    | 2           | BRAZIL    | 1           | y alongside of the pending deposits. carefully special packages are about the ironic forges. slyly special         |
    | 3           | CANADA    | 1           | eas hang ironic, silent packages. slyly regular packages are furiously over the tithes. fluffily bold              |
    | 4           | EGYPT     | 4           | y above the carefully unusual theodolites. final dugouts are quickly across the furiously regular d                |
    | 5           | ETHIOPIA  | 0           | ven packages wake quickly. regu                                                                                    |
    | 6           | FRANCE    | 3           | refully final requests. regular, ironi                                                                             |
    | 7           | GERMANY   | 3           | l platelets. regular accounts x-ray: unusual, regular acco                                                         |
    | 8           | INDIA     | 2           | ss excuses cajole slyly across the packages. deposits print aroun                                                  |
    | 9           | INDONESIA | 2           |  slyly express asymptotes. regular deposits haggle slyly. carefully ironic hockey players sleep blithely. carefull |
    | 10          | IRAN      | 4           | efully alongside of the slyly final dependencies.                                                                  |
    +-------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------+
    ");

    Ok(())
}

#[tokio::test]
async fn test_deregister_udtf() -> Result<()> {
    let ctx = SessionContext::new();

    ctx.register_udtf("read_csv", Arc::new(SimpleCsvTableFunc {}));

    assert!(ctx.state().table_functions().contains_key("read_csv"));

    ctx.deregister_udtf("read_csv");

    assert!(!ctx.state().table_functions().contains_key("read_csv"));

    Ok(())
}

#[derive(Debug)]
struct SimpleCsvTable {
    schema: SchemaRef,
    exprs: Vec<Expr>,
    batches: Vec<RecordBatch>,
}

#[async_trait]
impl TableProvider for SimpleCsvTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let batches = if !self.exprs.is_empty() {
            let max_return_lines = self.interpreter_expr(state).await?;
            // get max return rows from self.batches
            let mut batches = vec![];
            let mut lines = 0;
            for batch in &self.batches {
                let batch_lines = batch.num_rows();
                if lines + batch_lines > max_return_lines as usize {
                    let batch_lines = max_return_lines as usize - lines;
                    batches.push(batch.slice(0, batch_lines));
                    break;
                } else {
                    batches.push(batch.clone());
                    lines += batch_lines;
                }
            }
            batches
        } else {
            self.batches.clone()
        };
        Ok(MemorySourceConfig::try_new_exec(
            &[batches],
            TableProvider::schema(self),
            projection.cloned(),
        )?)
    }
}

impl SimpleCsvTable {
    async fn interpreter_expr(&self, state: &dyn Session) -> Result<i64> {
        use datafusion::logical_expr::expr_rewriter::normalize_col;
        use datafusion::logical_expr::utils::columnize_expr;
        let plan = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: true,
            schema: Arc::new(DFSchema::empty()),
        });
        let logical_plan = Projection::try_new(
            vec![columnize_expr(
                normalize_col(self.exprs[0].clone(), &plan)?,
                &plan,
            )?],
            Arc::new(plan),
        )
        .map(LogicalPlan::Projection)?;
        let rbs = collect(
            state.create_physical_plan(&logical_plan).await?,
            Arc::new(TaskContext::from(state)),
        )
        .await?;
        let limit = rbs[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        Ok(limit)
    }
}

#[derive(Debug)]
struct SimpleCsvTableFunc {}

impl TableFunctionImpl for SimpleCsvTableFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        let mut new_exprs = vec![];
        let mut filepath = String::new();
        for expr in exprs {
            match expr {
                Expr::Literal(ScalarValue::Utf8(Some(path)), _) => {
                    filepath.clone_from(path);
                }
                expr => new_exprs.push(expr.clone()),
            }
        }
        let (schema, batches) = read_csv_batches(filepath)?;
        let table = SimpleCsvTable {
            schema,
            exprs: new_exprs.clone(),
            batches,
        };
        Ok(Arc::new(table))
    }
}

/// Test that expressions passed to UDTFs are properly type-coerced
/// This is a regression test for https://github.com/apache/datafusion/issues/19914
#[tokio::test]
async fn test_udtf_type_coercion() -> Result<()> {
    use datafusion::datasource::MemTable;

    #[derive(Debug)]
    struct NoOpTableFunc;

    impl TableFunctionImpl for NoOpTableFunc {
        fn call_with_args(&self, _: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
            let schema = Arc::new(arrow::datatypes::Schema::empty());
            Ok(Arc::new(MemTable::try_new(schema, vec![vec![]])?))
        }
    }

    let ctx = SessionContext::new();
    ctx.register_udtf("f", Arc::new(NoOpTableFunc));

    // This should not panic - the array elements should be coerced to Float64
    let _ = ctx.sql("SELECT * FROM f(ARRAY[0.1, 1, 2])").await?;

    Ok(())
}

fn read_csv_batches(csv_path: impl AsRef<Path>) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    let mut file = File::open(csv_path)?;
    let (schema, _) = Format::default()
        .with_header(true)
        .infer_schema(&mut file, None)?;
    file.rewind()?;

    let reader = ReaderBuilder::new(Arc::new(schema.clone()))
        .with_header(true)
        .build(file)?;
    let mut batches = vec![];
    for batch in reader {
        batches.push(batch?);
    }
    let schema = Arc::new(schema);
    Ok((schema, batches))
}

/// A table function `pair(a, b)` with named parameters, whose one row holds
/// its arguments (`b` is optional and null when omitted).
#[derive(Debug)]
struct PairTableFunc {
    signature: Signature,
}

impl PairTableFunc {
    fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable)
                .with_parameter_names(vec!["a", "b"])
                .unwrap(),
        }
    }
}

impl TableFunctionImpl for PairTableFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        if args.exprs().len() > 2 {
            return plan_err!("pair takes at most two arguments");
        }
        let value = |i: usize| match args.exprs().get(i) {
            Some(Expr::Literal(ScalarValue::Int64(v), _)) => Ok(*v),
            None => Ok(None),
            Some(other) => plan_err!("pair: expected an integer, got {other}"),
        };
        let (a, b) = (value(0)?, value(1)?);
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![a])),
                Arc::new(Int64Array::from(vec![b])),
            ],
        )?;
        Ok(Arc::new(datafusion::datasource::MemTable::try_new(
            schema,
            vec![vec![batch]],
        )?))
    }

    fn signature(&self) -> Option<&Signature> {
        Some(&self.signature)
    }
}

/// The same function without a signature: positional arguments only.
#[derive(Debug)]
struct PositionalPairTableFunc;

impl TableFunctionImpl for PositionalPairTableFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        PairTableFunc::new().call_with_args(args)
    }
}

fn named_args_context() -> SessionContext {
    let ctx = SessionContext::new();
    ctx.register_udtf("pair", Arc::new(PairTableFunc::new()));
    ctx.register_udtf("positional_pair", Arc::new(PositionalPairTableFunc));
    ctx
}

async fn pair_rows(ctx: &SessionContext, sql: &str) -> Result<String> {
    let batches = ctx.sql(sql).await?.collect().await?;
    Ok(batches_to_string(&batches))
}

async fn pair_error(ctx: &SessionContext, sql: &str) -> String {
    ctx.sql(sql).await.unwrap_err().strip_backtrace()
}

const ONE_TWO: &str = "+---+---+\n| a | b |\n+---+---+\n| 1 | 2 |\n+---+---+";

#[tokio::test]
async fn test_udtf_named_arguments() -> Result<()> {
    let ctx = named_args_context();
    for sql in [
        "SELECT * FROM pair(1, 2)",
        "SELECT * FROM pair(a => 1, b => 2)",
        "SELECT * FROM pair(b => 2, a => 1)",
        "SELECT * FROM pair(1, b => 2)",
        "SELECT * FROM pair(A => 1, B => 2)",
        "SELECT * FROM positional_pair(1, 2)",
        "SELECT p.a, p.b FROM pair(b => 2, a => 1) AS p",
    ] {
        assert_eq!(pair_rows(&ctx, sql).await?, ONE_TWO, "{sql}");
    }

    // An omitted trailing parameter is not passed at all.
    assert_eq!(
        pair_rows(&ctx, "SELECT * FROM pair(a => 1)").await?,
        "+---+---+\n| a | b |\n+---+---+\n| 1 |   |\n+---+---+"
    );
    Ok(())
}

#[tokio::test]
async fn test_udtf_named_arguments_errors() -> Result<()> {
    let ctx = named_args_context();
    assert_contains!(
        pair_error(&ctx, "SELECT * FROM pair(c => 1)").await,
        "Unknown parameter name 'c'. Valid parameters are: [a, b]"
    );
    assert_contains!(
        pair_error(&ctx, "SELECT * FROM pair(a => 1, 2)").await,
        "Positional argument at position 1 follows named argument"
    );
    assert_contains!(
        pair_error(&ctx, "SELECT * FROM pair(1, a => 2)").await,
        "Parameter 'a' specified multiple times"
    );
    assert_contains!(
        pair_error(&ctx, "SELECT * FROM pair(b => 2)").await,
        "Missing required parameter 'a'"
    );
    assert_contains!(
        pair_error(&ctx, "SELECT * FROM pair(\"A\" => 1)").await,
        "Unknown parameter name 'A'"
    );
    assert_contains!(
        pair_error(&ctx, "SELECT * FROM positional_pair(a => 1)").await,
        "Table function 'positional_pair' does not support named arguments"
    );
    Ok(())
}

#[tokio::test]
async fn test_udtf_named_arguments_lateral() -> Result<()> {
    let ctx = named_args_context();
    assert_eq!(
        pair_rows(
            &ctx,
            "SELECT p.a, p.b FROM (VALUES (1)) AS t(x), LATERAL pair(b => 2, a => 1) AS p"
        )
        .await?,
        ONE_TWO
    );
    // Named arguments were once passed by position here, silently.
    assert_contains!(
        pair_error(
            &ctx,
            "SELECT * FROM (VALUES (1)) AS t(x), LATERAL positional_pair(b => 2, a => 1)"
        )
        .await,
        "Table function 'positional_pair' does not support named arguments"
    );
    Ok(())
}
