import asyncio
from agents import Agent, WebSearchTool, Runner
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

# ======= DEFINE THINKER AGENT =======

THINKER_INSTRUCTIONS = """
You are an expert researcher conducting deep research on a given topic in iterations.
Each iteration you gather more information, reflect on it and then decide on the next logical sub-topic to research.

OBJECTIVE:
Given a query, and the history of sub-topics and findings, provide your thoughts on the the next logical sub-topic to research.

GUIDELINES:
* The sub-topic you choose should be related to the original user query.
* Describe the next sub-topic in detail so that it can be researched by another agent.
* You should provide your thinking and justifications for the next topic based on what has been researched and discovered so far.
* If this is the first iteration, just provide your rationale for the first sub-topic to research, given that we do not have any other information yet.
"""

class ThinkerOutput(BaseModel):
    next_topic: str
    rationale: str

thinker = Agent(
    name="ThinkerAgent",
    instructions=THINKER_INSTRUCTIONS,
    output_type=ThinkerOutput,
    model="o3-mini"
)

# ======= DEFINE RESEARCHER AGENT =======

class ResearcherOutput(BaseModel):
    summary: str
    sources: List[str]

RESEARCHER_INSTRUCTIONS = """
You are an expert researcher conducting research on a given topic in iterations.

You are given:
* The original user query
* The specific sub-topic to be researched in this iteration
* The reasoning for investingating the sub-topic

OBJECTIVE:
Given the sub-topic, search the web for relevant information and summarize your findings as it relates to the sub-topic and original user query.

GUIDELINES:
* Use the WebSearchTool to search the web for relevant information.
* Ensure that the query you give the tool is well optimized SERP query for the sub-topic.
* Summarize your findings from the web search in 2-3 paragraphs and cite your sources inline.
"""

search_tool = WebSearchTool()

researcher = Agent(
    name="ResearcherAgent",
    instructions=RESEARCHER_INSTRUCTIONS,
    output_type=ResearcherOutput,
    model="gpt-4o-mini",
    tools=[search_tool]
)

# ======= DEFINE WRITER AGENT =======

WRITER_INSTRUCTIONS = """
You are an expert writer who is tasked with writing a report given a user query and a collection of findings put together by a team of researchers.

OBJECTIVE:
Write a comprehensive report that addresses the original user query based on the findings provided.
This should be as lengthy and detailed as possible given the available information.

GUIDELINES:
* Format the final report in markdown and use headings, sub-headings, bullets, and tables to organize information where appropriate.
* Cite your sources inline in the report.
"""

writer = Agent(
    name="WriterAgent",
    instructions=WRITER_INSTRUCTIONS,
    model="gpt-4o-mini"
)

# ======= CREATE THE RESEARCH LOOP =======


class DeepResearcher:
    def __init__(self):
        self.thoughts = []  # Thoughts produced by the thinker agent
        self.findings = []  # Findings produced by the researcher agent
        self.current_iteration = 1

    async def run(self, query: str, iterations: int = 4):

        while self.current_iteration <= iterations:

            thoughts = await self._do_thinking(query)
            findings = await self._do_research(query, thoughts)

            self.current_iteration += 1

        return await self._write_report(query)

    async def _do_thinking(self, query: str):

        historical_context = ""
        for i, thought in enumerate(self.thoughts):
            historical_context += f"ITERATION {i+1}:\n"
            historical_context += f"THOUGHT:\n{thought}\n"
            historical_context += f"FINDINGS:\n{self.findings[i]}\n\n"

        thinking_input = f"""
        User Query: {query}

        Previous Iterations:
        {historical_context}
        """

        thinker_response = await Runner.run(
            thinker,
            thinking_input
        )
        self.thoughts.append(thinker_response.final_output)

        thought = thinker_response.final_output
        print(f"========== ITERATION {self.current_iteration} ==========\n")
        print(f"Next Sub-Topic:\n{thought.next_topic}\n")
        print(f"Rationale:\n{thought.rationale}\n")
        return thought

    async def _do_research(self, query: str, thought: ThinkerOutput):
        research_input = f"""
        User Query: {query}
        
        Current Sub-Topic: {thought.next_topic}

        Rationale: {thought.rationale}
        """

        researcher_response = await Runner.run(
            researcher,
            research_input
        )
        self.findings.append(researcher_response.final_output)
        return researcher_response.final_output

    async def _write_report(self, query: str):
        findings_context = "\n\n".join(finding.summary for finding in self.findings)
        report_input = f"""
        User Query: {query}
 
        Research Findings:
        {findings_context}
        """
        
        writer_response = await Runner.run(
            writer,
            report_input
        )
        return writer_response.final_output


if __name__ == "__main__":
    # Run the deep research
    query = "Provide a deep-dive on the best platforms for video content creators to monetize their content and audience in 2025"
    iterations = 4
    deep_researcher = DeepResearcher()
    report = asyncio.run(deep_researcher.run(query, iterations))

    # Save the report to a markdown file
    file_name = "report.md"
    with open(file_name, "w") as f:
        f.write(report)

    print(f"Report saved to {file_name}")
