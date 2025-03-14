
```markdown
# Document Genie: Your AI-powered Insight Generator from PDFs

Document Genie is a powerful Streamlit application designed to extract and analyze text from PDF documents, leveraging the advanced capabilities of Huggingface's Generative AI, specifically the Gemini-PRO model. This tool uses a Retrieval-Augmented Generation (RAG) framework to offer precise, context-aware answers to user queries based on the content of uploaded documents.

## Features

- **Instant Insights**: Extracts and analyses text from uploaded PDF documents to provide instant insights.
- **Retrieval-Augmented Generation**: Utilizes Huggingface's Generative AI model Gemini-PRO for high-quality, contextually relevant answers.
- **Secure API Key Input**: Ensures secure entry of Huggingface API keys for accessing generative AI models.

## Getting Started

### Prerequisites

- Huggingface Access Token: Obtain a Huggingface Access Token to interact with Huggingface's Generative AI models. Visit [Huggingface API Key Setup](https://huggingface.co/settings/tokens) to get Access Token.
- Streamlit: This application is built with Streamlit. Ensure you have Streamlit installed in your environment.

### Installation

Clone this repository or download the source code to your local machine. Navigate to the application directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

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


