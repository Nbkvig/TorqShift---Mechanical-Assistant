# What & Why
 TorqShift, is an AI-powered web app designed for diy mechanics or car enthusiasts that may need fast and specific answers from their vehicle's owner's manual. Whether your in the middle of working on your car and can't remember the exact specs you need, TorqShift lets you ask a question and pull answers directly from official manuals (Currently only 2017 WRX and 2016 Odyssey). This allows you to get the exact answers you need without having to find your manual or going to many pages just to find what you need.

This app utilizes a rag pipeline that has both owner’s manual pre-processed, chunked, embedded, and stored in a local vector database (ChromaDB). When a user asks a question, the app retrieves the most relevant chunks within the vector database and passes them to GPT4o-mini to output a grounded answer. Users can also upload a photo of a component and the app will use GPT 4o-mini Vision to identify the parts before retrieving the information needed.

There are a couple of difficulties in getting the AI behavior right. First, the owner’s manuals are 500+ pages long with very dense information and tables. Meaning that some of the information are in a structured format that doesn’t quite chunk or embed that well when compared to just straight text. Second, hallucination could occur due to the tables, if the information was either lost or corrupted in a way when chunking / embedding. This could cause the LLM to infer the wrong information and output information that could harm users. The AI should only be grounding itself to the manual’s text and refuse to answer any questions where the context is insufficient. It should also not give general automotive information that isn’t directly stated in the manuals.

# Iterations:

 V1: Baseline

 Change: 
    Created the initial RAG pipeline with a hybrid PDF extraction using pdfplumber for tables, and PyMuPDF for regular text extraction. A basic system prompt that instructs the GPT 4o-mini to answer only from the retrieved manuals. Top_K is set to 5 chunks per query.

 Motivating Example: 
    Case 10 - "What is the top speed of the 2017 Subaru WRX?" scored only 0.25, it only matched the keyword "do not" out of the 4 expected keywords. The behavior of refusing to answer with hallucinated or misinformation was correct, but because the scoring is based off of it matching keywords in verbatim it failed. So, the score is showing bad, when in reality the response of the LLM was correct.

 Delta: 
    The Baseline mean score is ~87% across the 10 test cases

 Conclusion:
    The pipeline performed better than my expectations, with the scores on lookups for most fo the text cases were 1.00. The reason as to why I got lower scores was mostly due to the matching of the keywords, it doesn't take into account for the acutal response of the LLM it just looks at whether or not it matched the keywords. The next chagne would be to change up the test case for better measurement of the responses.



 V2: Imporved the System Prompt and added different eval keywrods

 Change: 
    I updated the system prompt in 'app.py' and 'eval/run_eval.py'. I added a instruction that will make the LLM to extract the numerical values, units and part names directlty from the pipeline when the answer contains specfic information (torq specs what not). I also updated the expected keywords for Case 10 which orignally only contained "not found" and "not in" to single words like "not", "cannot", "manual", and "excerpts" to measure the refusal behavior. Although the actual response is suppose to be right the test case score does not reflect that 

 Motivating Example: 
    Case 10 - "What is the top speed of the 2017 Subaru WRX?" scored only 0.25, it only matched the keyword "do not" out of the 4 expected keywords. The behavior of refusing to answer with hallucinated or misinformation was correct, but because the scoring is based off of it matching keywords in verbatim it failed. So, the score is showing bad, when in reality the response of the LLM was correct.

 Delta: 
    The Baseline mean score is ~87% across the 10 test cases

 Conclusion:
    The pipeline performed better than my expectations, with the scores on lookups for most fo the text cases were 1.00. The reason as to why I got lower scores was mostly due to the matching of the keywords, it doesn't take into account for the acutal response of the LLM it just looks at whether or not it matched the keywords. The next chagne would be to change up the test case for better measurement of the responses.
    


