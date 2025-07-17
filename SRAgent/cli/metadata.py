# import
## batteries
import sys
import asyncio
import argparse
from typing import List
## 3rd party
import pandas as pd
from Bio import Entrez
from langchain_core.messages import HumanMessage
## package
from SRAgent.cli.utils import CustomFormatter
from SRAgent.workflows.metadata import get_metadata_items, create_metadata_graph
from SRAgent.workflows.graph_utils import handle_write_graph_option
from SRAgent.agents.display import create_step_summary_chain
from SRAgent.db.get import db_get_unprocessed_records


# functions
def metadata_agent_parser(subparsers):
    help = 'Metadata Agent: Obtain metadata for specific SRX accessions'
    desc = """
    """
    sub_parser = subparsers.add_parser(
        'metadata', help=help, description=desc, formatter_class=CustomFormatter
    )
    sub_parser.set_defaults(func=metadata_agent_main)
    sub_parser.add_argument(
        'srx_accession_csv', type=str, nargs='?', default=None,
        help='CSV of entrez_id,srx_accession. Headers required. Optional when --from-db is used'
    )    
    sub_parser.add_argument(
        '--from-db', action='store_true', default=False,
        help='Get SRX accessions from database instead of CSV file'
    )
    sub_parser.add_argument(
        '--database', type=str, default='sra', choices=['gds', 'sra'], 
        help='Entrez database origin of the Entrez IDs'
    )
    sub_parser.add_argument(
        '--max-concurrency', type=int, default=6, help='Maximum number of concurrent processes'
    )
    sub_parser.add_argument(
        '--recursion-limit', type=int, default=200, help='Maximum recursion limit'
    )
    sub_parser.add_argument(
        '--max-parallel', type=int, default=2, help='Maximum parallel processing of SRX accessions'
    )
    sub_parser.add_argument(
        '--no-srr', action='store_true', default=False, 
        help='Do NOT upload SRR accessions to scBaseCount SQL database'
    )
    sub_parser.add_argument(
        '--use-database', action='store_true', default=False, 
        help='Add the results to the scBaseCount SQL database'
    )

    sub_parser.add_argument(
        '--write-graph', type=str, metavar='FILE', default=None,
        help='Write the workflow graph to a file and exit (supports .png, .svg, .pdf, .mermaid formats)'
    )
    sub_parser.add_argument(
        '--output-csv', type=str, default=None,
        help='Path to save the metadata as a CSV file'
    )
    sub_parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit the number of SRX accessions to process'
    )
    sub_parser.add_argument(
        '--filter-by', type=str, action='append', default=[],
        help='Filter results by key=value pairs (e.g., organism=Homo sapiens). Can be specified multiple times.'
    )
    sub_parser.add_argument(
        '--no-summaries', action='store_true', default=False,
        help='Do NOT print step summaries during processing'
    )


async def _process_single_srx(
    entrez_srx, database, graph, step_summary_chain, config: dict, no_summaries: bool
):
    """Process a single entrez_id"""
    # format input for the graph
    metadata_items = "\n".join([f" - {x}" for x in get_metadata_items().values()])
    prompt = "\n".join([
        "# Instructions",
        "For the SRA experiment accession {SRX_accession}, find the following dataset metadata:",
        metadata_items,
        "# Notes",
        " - Try to confirm any questionable metadata values with two data sources"
    ])
    input = {
        "entrez_id": entrez_srx[0],
        "SRX": entrez_srx[1],
        "database": database,
        "messages": [HumanMessage(prompt.format(SRX_accession=entrez_srx[1]))]
    }

    # call the graph
    final_state = None
    i = 0
    async for step in graph.astream(input, config=config):
        i += 1
        final_state = step
        if no_summaries:
            nodes = ",".join(list(step.keys()))
            print(f"[{entrez_srx[0]}] Step {i}: {nodes}")
        else:
            msg = await step_summary_chain.ainvoke({"step": step})
            print(f"[{entrez_srx[0]}] Step {i}: {msg.content}")

    extracted_metadata = {
        "entrez_id": entrez_srx[0],
        "SRX": entrez_srx[1],
        "is_illumina": None,
        "is_single_cell": None,
        "is_paired_end": None,
        "lib_prep": None,
        "tech_10x": None,
        "cell_prep": None,
        "organism": None,
        "tissue": None,
        "disease": None,
        "perturbation": None,
        "cell_line": None,
        "tissue_ontology_term_id": None,
        "SRR": None,
    }

    if final_state:
        try:
            extracted_metadata.update({
                "is_illumina": final_state.get("is_illumina"),
                "is_single_cell": final_state.get("is_single_cell"),
                "is_paired_end": final_state.get("is_paired_end"),
                "lib_prep": final_state.get("lib_prep"),
                "tech_10x": final_state.get("tech_10x"),
                "cell_prep": final_state.get("cell_prep"),
                "organism": final_state.get("organism"),
                "tissue": final_state.get("tissue"),
                "disease": final_state.get("disease"),
                "perturbation": final_state.get("perturbation"),
                "cell_line": final_state.get("cell_line"),
                "tissue_ontology_term_id": ", ".join(final_state.get("tissue_ontology_term_id", []) if final_state.get("tissue_ontology_term_id") is not None else []),
                "SRR": ", ".join(final_state.get("SRR", []) if final_state.get("SRR") is not None else []),
            })
            # Print final results after extracting, but ensure extracted_metadata is returned
            try:
                print(f"#-- Final results for Entrez ID {entrez_srx[0]} --#")
                print(final_state["final_state_node"]["messages"][-1].content)
                print("#---------------------------------------------#")
            except KeyError:
                print(f"#-- Final results for Entrez ID {entrez_srx[0]} --#")
                print("Could not retrieve final message content.")
                print("#---------------------------------------------#")
        except Exception as e:
            print(f"Error extracting metadata from final state: {e}")
            print("#---------------------------------------------#")
    return extracted_metadata

async def _metadata_agent_main(args):
    """
    Main function for invoking the metadata agent
    """
    try:
        print(f"Parsed arguments: {args}")
        
        # handle write-graph option
        if args.write_graph:
            handle_write_graph_option(create_metadata_graph, args.write_graph)
            return

        # create supervisor agent
        graph = create_metadata_graph()
        step_summary_chain = create_step_summary_chain()

        # invoke agent
        config = {
            "max_concurrency": args.max_concurrency,
            "recursion_limit": args.recursion_limit,
            "configurable": {
                "use_database": args.use_database,
                "no_srr": args.no_srr
            }
        }

        # read in entrez_id and srx_accession
        if args.from_db:
            from SRAgent.db.connect import db_connect
            from SRAgent.db.get import db_get_unprocessed_records, db_get_filtered_srx_metadata
            with db_connect() as conn:
                filters = {}
                for f in args.filter_by:
                    key, value = f.split('=', 1)
                    filters[key] = value

                # If filters are provided, use db_get_filtered_srx_metadata
                if filters:
                    records_df = db_get_filtered_srx_metadata(
                        conn=conn,
                        organism=filters.get('organism'),
                        is_single_cell=filters.get('is_single_cell'),
                        limit=args.limit,
                        database=args.database
                    )
                    entrez_srx_accessions = list(records_df[['entrez_id', 'srx_accession']].itertuples(index=False, name=None))
                else:
                    entrez_srx_accessions = db_get_unprocessed_records(conn, database=args.database)
                    # convert to list of tuples
                    entrez_srx_accessions = [(x, None) for x in entrez_srx_accessions]

                print(f"Retrieved records from DB: {entrez_srx_accessions}")
                if not entrez_srx_accessions:
                    print("No unprocessed records found in the database.")
                    return
                print("Reached end of db_connect block.")
        else:
            if not args.srx_accession_csv:
                raise ValueError("Either --from-db or srx_accession_csv must be provided.")
            # read in the entrez_id and srx_accession
            df_csv = pd.read_csv(args.srx_accession_csv)
            entrez_srx_accessions = list(df_csv[['entrez_id', 'srx_accession']].itertuples(index=False, name=None))

        # limit the number of SRX accessions to process
        if args.limit:
            entrez_srx_accessions = entrez_srx_accessions[:args.limit]

        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(args.max_parallel)

        async def _process_with_semaphore(entrez_srx):
            async with semaphore:
                return await _process_single_srx(
                    entrez_srx,
                    args.database,
                    graph,
                    step_summary_chain,
                    config,
                    args.no_summaries
                )

        # Process all SRX accessions concurrently
        raw_results = await asyncio.gather(*[_process_with_semaphore(esrx) for esrx in entrez_srx_accessions])
        valid_results = [res for res in raw_results if res is not None]

        if not valid_results:
            print("No metadata collected.")
            return

        all_srr_accessions = []
        for item in valid_results:
            if item.get("SRR"):
                srx_acc = item.get("SRX")
                srr_list = [s.strip() for s in item["SRR"].split(", ") if s.strip()]
                for srr in srr_list:
                    all_srr_accessions.append({"srx_accession": srx_acc, "srr_accession": srr})

        # Convert to DataFrame
        df = pd.DataFrame(valid_results)

        # Apply filters if provided (only for CSV input, DB filters handled by db_get_filtered_srx_metadata)
        if not args.from_db and args.filter_by:
            for filter_str in args.filter_by:
                try:
                    key, value = filter_str.split('=', 1)
                    if key in df.columns:
                        df = df[df[key].astype(str) == value] # Convert column to string for comparison
                    else:
                        print(f"Warning: Filter key '{key}' not found in metadata. Skipping filter.")
                except ValueError:
                    print(f"Warning: Invalid filter format '{filter_str}'. Expected 'key=value'. Skipping filter.")

        # Output results
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"Metadata saved to {args.output_csv}")
        else:
            print("\nCollected Metadata:")
            print(df.to_markdown(index=False))

        # Add to database
        if args.use_database:
            from SRAgent.db.upsert import db_upsert
            from SRAgent.db.utils import get_unique_columns
            with db_connect() as conn:
                # Upsert srx_metadata
                print(f"metadata_df shape: {df.drop(columns=['SRR']).shape}")
                print(f"metadata_df head:\n{df.drop(columns=['SRR']).head().to_string()}")
                db_upsert(df.drop(columns=["SRR"]), "srx_metadata", conn)
                print("SRX metadata upserted to database.")

                # Upsert srx_srr if --no-srr is not set
                if not args.no_srr and all_srr_accessions:
                    srr_df = pd.DataFrame(all_srr_accessions)
                    db_upsert(srr_df, "srx_srr", conn)
                    print("SRR accessions upserted to database.")


            try:
                key, value = filter_str.split('=', 1)
                if key in df.columns:
                    df = df[df[key].astype(str) == value] # Convert column to string for comparison
                else:
                    print(f"Warning: Filter key '{key}' not found in metadata. Skipping filter.")
            except ValueError:
                print(f"Warning: Invalid filter format '{filter_str}'. Expected 'key=value'. Skipping filter.")

        # Output results
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"Metadata successfully saved to {args.output_csv}")
        else:
            print("\nCollected Metadata:")
            print(df.to_markdown(index=False))

        # Add to database
        if args.use_database:
            from SRAgent.db.connect import db_connect
            from SRAgent.db.upsert import db_upsert
            from SRAgent.db.utils import get_unique_columns
            with db_connect() as conn:
                # Upsert srx_metadata
                unique_cols_metadata = get_unique_columns("srx_metadata", conn)
                print(f"Unique columns for srx_metadata: {unique_cols_metadata}")
                db_upsert(df.drop(columns=["SRR"]), "srx_metadata", conn)
                print("SRX metadata upserted to database.")

                # Upsert srx_srr if --no-srr is not set
                if not args.no_srr and all_srr_accessions:
                    srr_df = pd.DataFrame(all_srr_accessions)
                    unique_cols_srr = get_unique_columns("srx_srr", conn)
                    print(f"Unique columns for srr_srr: {unique_cols_srr}")
                    db_upsert(srr_df, "srx_srr", conn)
                    print("SRR accessions upserted to database.")

    except Exception as e:
        print(f"An error occurred: {e}")

def metadata_agent_main(args):
    print("Calling _metadata_agent_main...")
    asyncio.run(_metadata_agent_main(args))
    print("_metadata_agent_main call finished.")

# main
if __name__ == '__main__':
    pass