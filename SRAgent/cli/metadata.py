# import
## batteries
import os
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
        '--tenant', type=str, default='prod',
        choices=['prod', 'test'],
        help='Tenant name for the SRAgent SQL database'
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
    # set tenant
    if args.tenant:
        os.environ["DYNACONF"] = args.tenant

    # set email and api key
    Entrez.email = os.getenv("EMAIL")
    Entrez.api_key = os.getenv("NCBI_API_KEY")
    
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
        with db_connect() as conn:
            entrez_srx = db_get_unprocessed_records(conn)
            if not entrez_srx:
                print("No unprocessed records found in the database.")
                return
    elif args.srx_accession_csv:
        entrez_srx = pd.read_csv(args.srx_accession_csv, comment="#").to_records(index=False)
    else:
        print("Error: Either --from-db or srx_accession_csv must be provided.")
        return

    # Apply limit if specified
    if args.limit is not None:
        entrez_srx = entrez_srx[:args.limit]

    # Create semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(args.max_parallel)

    async def _process_with_semaphore(entrez_id):
        async with semaphore:
            return await _process_single_srx(
                entrez_id,
                args.database,
                graph,
                step_summary_chain,
                config,
                args.no_summaries
            )

    # Process all SRX accessions concurrently
    raw_results = await asyncio.gather(*[_process_with_semaphore(esrx) for esrx in entrez_srx])
    valid_results = [res for res in raw_results if res is not None]

    if not valid_results:
        print("No metadata collected.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(valid_results)

    # Apply filters if specified
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
        print(f"Metadata successfully saved to {args.output_csv}")
    else:
        print("\nCollected Metadata:")
        print(df.to_markdown(index=False))

def metadata_agent_main(args):
    asyncio.run(_metadata_agent_main(args))

# main
if __name__ == '__main__':
    pass