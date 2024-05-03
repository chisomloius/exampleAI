from llama_index.core import PromptTemplate

instruction_str = """\
                1. Covert the query to executable Python code using Pandas.
                2. The final line of code should be a python expression that can be called with the `eval()` function.
                3. the code should represent a solution to the query.
                4. Print only the expression.
                5.Don't quote the expression."""


new_prompt = PromptTemplate(
        """\
        You are wokring with a pandas dataframe in Python.
        The name of the dataframe is `df`,
        This is the result of `print(df.head())`: {df_str}

        Follow these instructions:
        {instruction_str}
        Query: {query_str}

        Expression: """
)


context = """Purpose: The primary role of this agent is to assist users by providing accurate information about world population statistics and details about a country. """
