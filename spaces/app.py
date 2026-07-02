"""Gradio demo for Meridian: agentic RAG over a domain-specific document corpus.

Wraps the compiled Meridian graph in a Gradio interface. Reads all
credentials from environment variables (GROQ_API_KEY, TAVILY_API_KEY,
QDRANT_URL, QDRANT_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY); none
are hardcoded here. Configure them as Space secrets before deployment.
"""

import uuid

import gradio as gr

from meridian.graph.graph import run_query


def _format_sources(graded_doc_list: list[dict]) -> str:
    """Render graded documents as a markdown list of titles and scores."""
    if not graded_doc_list:
        return "(no corpus documents passed the relevance threshold)"
    line_list = []
    for doc in graded_doc_list:
        payload_dict = doc.get("payload", {})
        title = payload_dict.get("title", "(untitled)")
        score = doc.get("rerank_score", 0.0)
        line_list.append(f"- **{title}** (score: {score:.3f})")
    return "\n".join(line_list)


def _format_graph_path(final_state: dict) -> str:
    """Render a summary of which recovery paths fired for this query."""
    part_list = [f"query_type: {final_state.get('query_type', 'unknown')}"]
    source = final_state.get("source", "corpus")
    part_list.append(f"source: {source}")
    if source == "web":
        part_list.append("route: grade_documents -> web_search -> generate")
    else:
        part_list.append("route: grade_documents -> generate")
    if final_state.get("hallucination_score") == "yes":
        part_list.append("recovery: hallucination detected, generation retried")
    if final_state.get("answer_score") == "no":
        part_list.append("recovery: off-target answer, query rewritten and re-retrieved")
    part_list.append(f"iteration_count: {final_state.get('iteration_count', 0)}")
    return "\n".join(part_list)


def answer_query(query: str, session_id: str) -> tuple[str, str, str]:
    """Run the Meridian graph for a query and return answer, sources, and path.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    session_id : str
        Cross-session memory key. A random id is generated if left blank.

    Returns
    -------
    tuple of str
        The generated answer, a markdown list of retrieved sources, and a
        summary of the graph path taken.
    """
    if not query.strip():
        return "Enter a question first.", "", ""
    resolved_session_id = session_id.strip() or str(uuid.uuid4())
    final_state = run_query(query, thread_id=resolved_session_id, session_id=resolved_session_id)
    answer = final_state.get("generation", "")
    sources = _format_sources(final_state.get("graded_docs", []))
    graph_path = _format_graph_path(final_state)
    return answer, sources, graph_path


with gr.Blocks(title="Meridian: Agentic RAG") as demo:
    gr.Markdown(
        "# Meridian\n"
        "Agentic retrieval-augmented generation with hybrid retrieval, "
        "CRAG-style document grading, and grounded, cited responses."
    )
    with gr.Row():
        query_input = gr.Textbox(label="Question", placeholder="Ask a question...", scale=4)
        session_input = gr.Textbox(
            label="Session ID (optional)",
            placeholder="leave blank for a new session",
            scale=2,
        )
    submit_button = gr.Button("Submit", variant="primary")
    answer_output = gr.Textbox(label="Answer", lines=6)
    with gr.Row():
        sources_output = gr.Markdown(label="Retrieved sources")
        path_output = gr.Textbox(label="Graph path taken", lines=6)

    submit_button.click(
        fn=answer_query,
        inputs=[query_input, session_input],
        outputs=[answer_output, sources_output, path_output],
    )

if __name__ == "__main__":
    demo.launch()
