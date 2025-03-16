
# Financial AI Assistant: Get Qualcomm's quarterly earnings
## Hosted Website : https://rag-finance-chatbot-xebqx6exkcs3fffjrtxryz.streamlit.app/

## How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need API key for the chatbot to access Huggingface's Generative AI models. Obtain your Access Token https://huggingface.co/settings/tokens.

3. **Ask a Question**: Ask any questions related to Qualcomm's earnings from 2023 to 2025.

## Getting Started

### Prerequisites

- Huggingface Access Token: Obtain a Huggingface Access Token to interact with Huggingface's Generative AI models. Visit [Huggingface API Key Setup](https://huggingface.co/settings/tokens) to get Access Token.
- Streamlit: This application is built with Streamlit. Ensure you have Streamlit installed in your environment.

### Installation

Clone this repository or download the source code to your local machine. Navigate to the application directory and install the required Python packages:

pip install -r requirements.txt


### How to Use

1. **Start the Application**: Launch the Streamlit application by running the command:
    ```bash
    streamlit run <path_to_script.py>
    ```
    Replace `<path_to_script.py>` with the path to the script file.

2. **Enter Your Huggingface Access Token**: Securely enter your Huggingface Access Token when prompted. This token enables the application to access Huggingface's Generative AI models.

3. **Ask Questions**: Ask any questions related to Qualcomm's earnings from 2023 to 2025.

### Technical Overview

- **PDF Processing**: Utilizes `PyPDFDirectoryLoader` for extracting text from PDF documents.
- **Text Chunking**: Employs the `RecursiveCharacterTextSplitter` from LangChain for dividing the extracted text into manageable chunks.
- **Vector Store Creation**: Uses `Chroma` for creating a searchable vector store from text chunks.
- **Answer Generation**: Leverages `HuggingFaceHub` from LangChain for generating answers to user queries using the context provided by the uploaded documents.


### Support

For issues, questions, or contributions, please refer to the GitHub repository issues section or submit a pull request.


