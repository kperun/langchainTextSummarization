import os
import dotenv
from langchain_openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import streamlit as st

# import streamlit as st

# Load the .env file
dotenv.load_dotenv('API.env')
OpenAI.api_key = os.getenv("OPEN_API_KEY")


def test_open_file(file_url):
    """
    Loads and splits the PDF file into pages, then prints the number of pages and the content of the first page.
    If the document has no pages, it exits the program.
    """
    loader = PyPDFLoader(file_url)
    pages = loader.load_and_split()

    # check if document loading was correct
    if pages:
        print('Number of pages: ', len(pages))
        print(pages[0].page_content)
    else:
        print('Document with 0 pages loaded, exit')
        exit(0)


def summarize_pdf(i_file_path, i_chunk_size, i_overlap, i_prompt_template=None, i_map_prompt_template=None,
                  i_reduce_prompt_template=None,
                  i_chain_type='stuff'):
    """
    Summarizes the content of a PDF file.

    Parameters:
    - file_path (str): The path to the PDF file.
    - chunk_size (int): The size of the text chunks to split the document into.
    - overlap (int): The number of overlapping characters between chunks.
    - chain_type(str): Either stuff or map_reduce, default is stuff.
    Returns:
    - str: The summary of the first chunk of the document.
    """
    # Instantiate the model
    llm = ChatOpenAI(model='gpt-3.5-turbo-16k', temperature=0, openai_api_key=OpenAI.api_key)
    # Load the file
    loader = PyPDFLoader(i_file_path)
    docs_raw = loader.load()

    # Create multiple documents
    docs_raw_text = [doc.page_content for doc in docs_raw]

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=i_chunk_size, chunk_overlap=i_overlap)
    docs_chunks = text_splitter.create_documents(docs_raw_text)

    # summarize the chunks
    if i_map_prompt_template and i_reduce_prompt_template:
        chain = load_summarize_chain(llm, chain_type=i_chain_type, map_prompt=i_map_prompt_template,
                                     combine_prompt=i_reduce_prompt_template)
    elif i_prompt_template:
        chain = load_summarize_chain(llm, chain_type=i_chain_type, prompt=i_prompt_template)
    else:
        chain = load_summarize_chain(llm, chain_type=i_chain_type)

    # return the summary
    summary = chain.invoke(docs_chunks, return_only_outputs=True)
    return summary['output_text']


def run_example():
    # Load the PDF file which will be used for RAG
    pdf_url = "https://www.medrxiv.org/content/10.1101/2021.07.15.21260605v1.full.pdf"
    # Print the summary of the PDF file
    print(summarize_pdf(pdf_url, 1000, 20, i_chain_type='stuff'))
    # We can add a prompt template to achieve specific results
    prompt_template = """
                        Write a summary of the document which includes the main points and any important details.
                        Highlight the 3 most important findings as bullet points.
                        Input document:
                        {text}    

                        """
    # initialize it
    prompt = PromptTemplate(
        input_variable=['text'],
        template=prompt_template
    )
    print(summarize_pdf(pdf_url, 1000, 20, i_chain_type='stuff', i_prompt_template=prompt))
    # Now with map reduce approach with appropriate prompts:
    # Create the prompt template for the map phase
    map_prompt_template = """
                                Write a summary of the document which includes the main points and any important details.
                                Highlight the 3 most important findings as bullet points.
                                Input document:
                                {text}    

                                """
    # initialize it
    map_prompt = PromptTemplate(
        input_variable=['text'],
        template=map_prompt_template
    )
    # Create the prompt template for the reduce phase
    reduce_prompt_template = """
                                    You will be given the main points and any important details of a research paper as 
                                    a text and in bullet points. Your goal is to give a final summary as a short text and
                                    bullet points.
                                    {text}    
                                    Final Summary:
                                    """
    # initialize it
    reduce_prompt = PromptTemplate(
        input_variable=['text'],
        template=reduce_prompt_template
    )
    print(summarize_pdf(pdf_url, 1000, 20, i_map_prompt_template=map_prompt,
                        i_reduce_prompt_template=reduce_prompt, i_chain_type='map_reduce'))


def main():
    st.set_page_config(page_title="PDF Summarizer", page_icon=":book", layout="wide")
    st.title("PDF Summarizer")
    # Get the input pdf file to summarize
    pdf_file_path = st.text_input("Path to PDF:")
    if pdf_file_path != '':
        st.write('PDF loaded!')
    else:
        exit(0)
    # Overlap and chunk size
    chunk_size = st.sidebar.number_input(label="Chunk size", min_value=200, max_value=5000)
    overlap = st.sidebar.number_input(label="Overlap", min_value=10, max_value=100)


    # Chain type
    method = st.sidebar.selectbox('Chain type:', ('stuff', 'map_reduce'))
    # Prompt
    if method == 'stuff':
        # Prompt Input
        user_prompt = st.text_input("Your prompt:")
        user_prompt = user_prompt + """{text}"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template=user_prompt
        )
    else:
        map_prompt_template = st.text_input("Map prompt:")
        map_prompt_template = map_prompt_template + """{text}"""
        map_prompt = PromptTemplate(
            input_variables=["text"],
            template=map_prompt_template
        )
        # Reduce prompt
        reduce_prompt_template = st.text_input("Reduce prompt:")
        reduce_prompt_template = reduce_prompt_template + """{text}"""
        reduce_prompt = PromptTemplate(
            input_variables=["text"],
            template=reduce_prompt_template
        )

    #
    if st.button("Summarize"):
        if not chunk_size or not overlap or not method:
            st.write("Please provide valid arguments!")

        if method == 'stuff':
            summary = summarize_pdf(pdf_file_path, chunk_size, overlap, i_chain_type=method,
                                    i_prompt_template=prompt)
            st.write(summary)
        else:
            summary = summarize_pdf(pdf_file_path, chunk_size, overlap,
                                    i_chain_type=method,
                                    i_map_prompt_template=map_prompt,
                                    i_reduce_prompt_template=reduce_prompt)
            st.write(summary)


if __name__ == '__main__':
    main()
