import PyPDF2
from transformers import pipeline
from pydantic import PrivateAttr  # Import PrivateAttr for private fields
from .basetool import BaseTool

class PDFReaderTool(BaseTool):
    _summarizer: pipeline = PrivateAttr()  # Private attribute for the summarizer

    def __init__(self, model_name: str = 'facebook/bart-large-cnn'):
        """
        Initializes the summarization pipeline with a pre-trained model and sets the tool name and description.
        
        :param model_name: Name of the pre-trained summarization model.
        """
        # Initialize the BaseTool with name and description
        super().__init__(
            name="pdf_reader_tool",
            description="A tool for reading and summarizing PDF files."
        )
        
        # Now initialize the summarizer
        self._initialize_summarizer(model_name)

    def _initialize_summarizer(self, model_name: str):
        """Initializes the summarizer model pipeline."""
        try:
            self._summarizer = pipeline("summarization", model=model_name)
        except Exception as e:
            print(f"Error loading summarization model: {str(e)}")
            self._summarizer = None

    def read_pdf(self, file_path: str) -> str:
        """
        Reads the text from a PDF file.

        :param file_path: Path to the PDF file.
        :return: Extracted text from the PDF file.
        """
        try:
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            return f"Error reading PDF file: {str(e)}"

    def summarize_pdf(self, file_path: str, max_length: int = 200, min_length: int = 50) -> str:
        """
        Summarizes the content of a PDF file using a transformer model.

        :param file_path: Path to the PDF file.
        :param max_length: Maximum length of the summary.
        :param min_length: Minimum length of the summary.
        :return: Summary of the PDF content.
        """
        if not hasattr(self, '_summarizer') or self._summarizer is None:
            return "Summarization model is not loaded."

        try:
            # Extract text from PDF
            text = self.read_pdf(file_path)

            # Handle long texts by splitting into manageable chunks
            max_chunk_size = 1000  # Adjust based on model's max input length
            text_chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

            # Summarize each chunk and combine the summaries
            summaries = []
            for chunk in text_chunks:
                summary = self._summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                )[0]['summary_text']
                summaries.append(summary)

            # Combine summaries
            final_summary = ' '.join(summaries)

            return final_summary
        except Exception as e:
            return f"Error summarizing PDF file: {str(e)}"

    def _run(self, file_path: str = None, **kwargs) -> str:
        """
        Synchronous run method that reads the PDF content.
        
        :param file_path: Path to the PDF file.
        :return: The full content of the PDF file.
        """
        # Handle the case where the argument is passed as "__arg1" or other names
        if not file_path:
            file_path = kwargs.get('__arg1')

        if not file_path:
            return "Error: No file path provided."

        return self.read_pdf(file_path)



    async def _arun(self, file_path: str) -> str:
        """
        Asynchronous run method that summarizes a PDF file.

        :param file_path: Path to the PDF file.
        :return: Summary of the PDF content.
        """
        return self.summarize_pdf(file_path)
