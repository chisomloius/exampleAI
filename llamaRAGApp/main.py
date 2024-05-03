from dotenv import load_dotenv
import os
import pandas as pd


# from llama_index.core.query_engine import PandasQueryEngine
from llama_index.experimental.query_engine import PandasQueryEngine
from prompt import new_prompt, instruction_str, context


from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import israel_engine

load_dotenv()


path  = os.path.join("data", "population.csv")
df = pd.read_csv(path)
# print(df.head())


population_query_engine = PandasQueryEngine(df=df, verbose=True, instruction_str=instruction_str)
population_query_engine.update_prompts({"pandas_prompt" : new_prompt})
# populationQueryEngine.query("What is the population of israel")


tools =[
    note_engine,
    QueryEngineTool(query_engine=population_query_engine, 
                    metadata=ToolMetadata(
                        name="population_data",
                        description="this gives information at the world population and demographics",
                    )
    ),
    QueryEngineTool(query_engine=israel_engine, 
                    metadata=ToolMetadata(
                        name="israel_data",
                        description="this gives detailed information about the country of israel",
                    )
    )
]


llm = OpenAI(model="gpt-3.5-turbo-0613")
agent =  ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)


while(prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)

    print(result)