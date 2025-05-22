import os
from pdfminer.high_level import extract_text as pdf_extract_text
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed # More specific exception for encrypted/protected PDFs
from pdfminer.psparser import PSError # For other parsing errors

from langchain_community.text_splitters import RecursiveCharacterTextSplitter
# Optional: If using tiktoken for length calculation
# import tiktoken

# --- Text Extraction Functions ---

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text content from a PDF file using pdfminer.six.

    Args:
        file_path: The path to the PDF file.

    Returns:
        The extracted text as a string.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        PDFTextExtractionNotAllowed: If text extraction is not allowed from the PDF.
        Exception: For other errors during PDF processing (e.g., corrupt file).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    try:
        print(f"Attempting to extract text from PDF: {file_path}")
        text = pdf_extract_text(file_path)
        print(f"Successfully extracted text from PDF: {file_path}. Length: {len(text)}")
        return text
    except FileNotFoundError: # Should be caught by the os.path.exists check, but good to have
        raise
    except PDFTextExtractionNotAllowed:
        print(f"Error: Text extraction not allowed from PDF: {file_path}")
        raise PDFTextExtractionNotAllowed(f"Text extraction is not allowed from '{file_path}'. The PDF might be encrypted or protected.")
    except PSError as e: # Catch pdfminer's PostScript errors (often due to malformed PDFs)
        print(f"Error processing PDF (PSError) '{file_path}': {e}")
        raise Exception(f"Failed to process PDF file '{file_path}' due to a parsing error (PSError): {e}")
    except Exception as e:
        # Catch other potential errors from pdfminer.six or unexpected issues
        print(f"An unexpected error occurred while extracting text from PDF '{file_path}': {e}")
        raise Exception(f"An unexpected error occurred while processing PDF file '{file_path}': {e}")


def extract_text_from_txt(file_path: str) -> str:
    """
    Extracts text content from a TXT file.

    Args:
        file_path: The path to the TXT file.

    Returns:
        The extracted text as a string.

    Raises:
        FileNotFoundError: If the TXT file does not exist.
        Exception: For other errors during file reading.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TXT file not found at: {file_path}")

    try:
        print(f"Attempting to extract text from TXT: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Successfully extracted text from TXT: {file_path}. Length: {len(text)}")
        return text
    except FileNotFoundError: # Should be caught by os.path.exists
        raise
    except UnicodeDecodeError as e:
        print(f"Encoding error while reading TXT file '{file_path}': {e}. Trying with 'latin-1'.")
        try:
            with open(file_path, 'r', encoding='latin-1') as f: # Fallback encoding
                text = f.read()
            print(f"Successfully extracted text from TXT with latin-1: {file_path}. Length: {len(text)}")
            return text
        except Exception as e_fallback:
            print(f"Fallback encoding failed for TXT file '{file_path}': {e_fallback}")
            raise Exception(f"Failed to read TXT file '{file_path}' due to encoding issues: {e_fallback}")
    except Exception as e:
        print(f"An error occurred while reading TXT file '{file_path}': {e}")
        raise Exception(f"An error occurred while reading TXT file '{file_path}': {e}")


# --- Text Splitting Function ---

def split_text_into_chunks(
    text: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200,
    use_tiktoken: bool = False # Optional: flag to use tiktoken based splitting
) -> list[str]:
    """
    Splits a given text into smaller chunks using LangChain's text splitters.

    Args:
        text: The text content to be split.
        chunk_size: The maximum size of each chunk (in characters or tokens).
        chunk_overlap: The number of characters or tokens to overlap between chunks.
        use_tiktoken: If True, uses tiktoken for chunk size calculation (more model-aware). 
                      Otherwise, uses character count.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    if use_tiktoken:
        # Example: Using tiktoken for length calculation (requires tiktoken library)
        # try:
        #     # TODO: Ensure 'tiktoken' is in requirements.txt if this path is enabled
        #     # This assumes a model like gpt-3.5-turbo or gpt-4. Adjust model name if needed.
        #     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        #         model_name="gpt-3.5-turbo", # Or another model name relevant to your LLM
        #         chunk_size=chunk_size,      # Now in tokens
        #         chunk_overlap=chunk_overlap # Now in tokens
        #     )
        #     print(f"Splitting text using tiktoken encoder. Target chunk size: {chunk_size} tokens, overlap: {chunk_overlap} tokens.")
        # except ImportError:
        #     print("Warning: tiktoken library not found. Falling back to character-based splitting.")
        #     text_splitter = RecursiveCharacterTextSplitter(
        #         chunk_size=chunk_size * 4, # Rough approximation: 1 token ~ 4 chars
        #         chunk_overlap=chunk_overlap * 4,
        #         length_function=len,
        #         add_start_index=True, # Optional: adds start index to metadata
        #     )
        # For now, let's stick to character-based to avoid direct tiktoken import unless explicitly enabled
        # and to simplify testing if tiktoken isn't installed by default in test env.
        # The subtask says "basic RecursiveCharacterTextSplitter is fine".
        print("Tiktoken-based splitting requested but not fully implemented in this version. Falling back to character-based.")
        # Fall-through to character-based for this example until fully wired.
        # To properly enable, the try-except block above should be used.

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # In characters if not using from_tiktoken_encoder
        chunk_overlap=chunk_overlap, # In characters
        length_function=len,
        add_start_index=False, # Optional: adds start index to metadata of Document objects
                              # For list[str] output, this isn't directly used.
    )
    
    print(f"Splitting text using character count. Chunk size: {chunk_size} chars, overlap: {chunk_overlap} chars.")
    chunks = text_splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks.")
    return chunks


if __name__ == '__main__':
    # Example Usage (Illustrative - requires actual files to run)
    print("--- Document Processor Example ---")

    # Create dummy files for testing
    dummy_txt_content = "This is a test text file.\nIt has multiple lines.\n" + "L" * 1500 # Long line
    dummy_pdf_file = "dummy_test.pdf" # Needs a real PDF to work
    dummy_txt_file = "dummy_test.txt"

    with open(dummy_txt_file, "w", encoding="utf-8") as f:
        f.write(dummy_txt_content)

    # Note: PDF creation is complex. For a real test, provide a sample PDF file.
    # For now, PDF extraction will likely fail or be skipped if dummy_test.pdf doesn't exist or is invalid.
    print(f"\nTo test PDF extraction, create a file named '{dummy_pdf_file}' in the same directory.")

    # 1. Test TXT Extraction
    print(f"\n--- Testing TXT Extraction ({dummy_txt_file}) ---")
    try:
        txt_extracted_text = extract_text_from_txt(dummy_txt_file)
        print(f"Successfully extracted from TXT. Length: {len(txt_extracted_text)}")
        # print(f"Content:\n{txt_extracted_text[:200]}...") # Print snippet
    except Exception as e:
        print(f"Error extracting from TXT: {e}")

    # 2. Test PDF Extraction (if dummy_test.pdf exists and is valid)
    print(f"\n--- Testing PDF Extraction ({dummy_pdf_file}) ---")
    if os.path.exists(dummy_pdf_file):
        try:
            pdf_extracted_text = extract_text_from_pdf(dummy_pdf_file)
            print(f"Successfully extracted from PDF. Length: {len(pdf_extracted_text)}")
            # print(f"Content:\n{pdf_extracted_text[:200]}...") # Print snippet
            
            # 3. Test Text Splitting (using PDF content if available, else TXT)
            text_to_split = pdf_extracted_text if pdf_extracted_text else (txt_extracted_text if 'txt_extracted_text' in locals() else "")
            if text_to_split:
                print("\n--- Testing Text Splitting ---")
                chunks = split_text_into_chunks(text_to_split, chunk_size=500, chunk_overlap=50)
                print(f"Split into {len(chunks)} chunks.")
                for i, chunk in enumerate(chunks):
                    print(f"  Chunk {i+1} (length {len(chunk)}): {chunk[:80]}...")
            else:
                print("\nSkipping text splitting as no text was extracted.")

        except Exception as e:
            print(f"Error extracting from PDF: {e}")
    else:
        print(f"'{dummy_pdf_file}' not found. Skipping PDF extraction test.")
        # If PDF skipped, try splitting TXT content
        if 'txt_extracted_text' in locals() and txt_extracted_text:
            print("\n--- Testing Text Splitting (using TXT content) ---")
            chunks = split_text_into_chunks(txt_extracted_text, chunk_size=500, chunk_overlap=50)
            print(f"Split into {len(chunks)} chunks.")
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i+1} (length {len(chunk)}): {chunk[:80]}...")


    # Clean up dummy file
    if os.path.exists(dummy_txt_file):
        os.remove(dummy_txt_file)
    
    print("\n--- Document Processor Example Finished ---")
