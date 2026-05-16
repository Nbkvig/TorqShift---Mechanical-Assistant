# What & Why
 TorqShift, is an AI-powered web app designed for diy mechanics or car enthusiasts that may need fast and specific answers from their vehicle's owner's manual. Whether your in the middle of working on your car and can't remember the exact specs you need, TorqShift lets you ask a question and pull answers directly from official manuals (Currently only 2017 WRX and 2016 Odyssey). This allows you to get the exact answers you need without having to find your manual or going to many pages just to find what you need.

This app utilizes a rag pipeline that has both owner’s manual pre-processed, chunked, embedded, and stored in a local vector database (ChromaDB). When a user asks a question, the app retrieves the most relevant chunks within the vector database and passes them to GPT4o-mini to output a grounded answer. Users can also upload a photo of a component and the app will use GPT 4o-mini Vision to identify the parts before retrieving the information needed.

There are a couple of difficulties in getting the AI behavior right. First, the owner’s manuals are 500+ pages long with very dense information and tables. Meaning that some of the information are in a structured format that doesn’t quite chunk or embed that well when compared to just straight text. Second, hallucination could occur due to the tables, if the information was either lost or corrupted in a way when chunking / embedding. This could cause the LLM to infer the wrong information and output information that could harm users. The AI should only be grounding itself to the manual’s text and refuse to answer any questions where the context is insufficient. It should also not give general automotive information that isn’t directly stated in the manuals.

# Iterations:

 ## V1: Baseline

 Change: 
    Created the initial RAG pipeline with a hybrid PDF extraction using pdfplumber for tables, and PyMuPDF for regular text extraction. A basic system prompt that instructs the GPT 4o-mini to answer only from the retrieved manuals. Top_K is set to 5 chunks per query.

 Motivating Example: 
    Case 10 - "What is the top speed of the 2017 Subaru WRX?" scored only 0.25, it only matched the keyword "do not" out of the 4 expected keywords. The behavior of refusing to answer with hallucinated or misinformation was correct, but because the scoring is based off of it matching keywords in verbatim it failed. So, the score is showing bad, when in reality the response of the LLM was correct.

 Delta: 
    The Baseline mean score is ~87% across the 10 test cases

 Conclusion:
    The pipeline performed better than my expectations, with the scores on lookups for most fo the text cases were 1.00. The reason as to why I got lower scores was mostly due to the matching of the keywords, it doesn't take into account for the acutal response of the LLM it just looks at whether or not it matched the keywords. The next change would be to change up the test case for better measurement of the responses.



 ## V2: Improved the System Prompt and added different eval keywords

 Change: 
    I updated the system prompt in 'app.py' and 'eval/run_eval.py'. I added a instruction that will make the LLM to extract the numerical values, units and part names directlty from the pipeline when the answer contains specfic information. I also updated the expected keywords for Case 10 which orignally only contained "not found" and "not in" to single words like "not", "cannot", "manual", and "excerpts" to measure the refusal behavior. Although the actual response is suppose to be right the test case score does not reflect that 

 Motivating Example: 
    Case 10 -- "What is the top speed of the 2017 Subaru WRX?" scored only 0.25 in v1 despite the model correctly refusing to answer. The expected keywords were not singluar words so it made it harder to match to the reponses as it was expecting it to match for verbatim. The model's score shouldn't be penalized if the response was right.

 Delta: 
    V1: 0.87
    V2: 0.87
    There was no overall change. Case 10 did improve from 0.25 to 0.50 with adding some of the new keywords as stated in the change section. However, case 4 did change from 1.0 to 0.75 as the new updated prompt caused the model to drop the quart from the response and most likely abbreviated it due the the offical manuals. 
    

 Conclusion:
    The prompt change and keyword kinda fixed the scoring, as with case 10 saw improvements but case 4's score dropped due to the non-deterministic results and system prompts. So the next course of action would be to change the temperature's score to 0.0 to make it more deterministic, since I forgot to add that. Considering that case 10 is still scored wrongly I'll change it in the V4 most likely to maybe a LLM to judge the responses.
    



 ## V3: Added Temperature 0.0 and changed the Top_k from 5 --> 8

 Change: 
    I set the temperature for GPT4o-mini responses to make them more deterministic and consistent ouputs. I also increased the TOP_K from 5 to 8 to allow the LLM to retrieve more context chunks to allow for it to create a better and more accurate response from my perspective.

 Motivating Example: 
    Case 4 -- "What is the coolant capacity for the 2017 WRX?" was scored worse in V2 than that of v1's run. V1 was 1.0 while V2 was 0.75, after looking at the results it was due to the output response changing "quart" to abbreviated verion of the unit (quart --> qt). Between these two runs the inconsistency was due to non deterministic outputs, so when running between the two runs it fluctuated the score.

 Delta: 
    V2: 0.87
    V3: 0.95
    There was a 8% improvment from version 2 to 3 with the changes made. C section. After adding the temperature and changing the top_k it made case 4's score return back to 1.0. Although, it wasn't mentioned in the previous version case 7 was at 0.67 in V2 and it also went back up to 1.0. Case 10 also had an improvement from 0.50 to 0.75.
    

 Conclusion:
    By adding the temperature to 0.0 helped eliminate the LLM's non deterministic responses, which helped in proving more of a consistent response and aligned itself to the keywords. Increasing the Top_k helped give the LLM some more context, by allowing it to retrieve 8 chunks rather than 5. Providing that more context helped case 7 with the step-by-step response. Just like previously stated back in V2, the reason why case 10's score still is the same is due to the test cases only measuring based off of the matching keywords, so v4 will be to add a LLM to judge the response to reliably measuree it.



 ## V4: LLM as a Judge

 Change:
    Replaced the keyword matching measure within 'eval/run_eval.py' with a llm acting as a judge scoring the responses. So, now each response should now be scored by the model 4o-mini acting as the judge. The model will receive the original query and the expected test case notes as well as the model's response then scores the correctness of the response from 0.0 to 1.0.
    

 Motivating Example: 
    case 10 -- "What is the top speed of the 2017 Subaru WRX?" score was always showing a below 1.00 score across previous iterations. The keyword matching cannot evaluate the actual context of the generated response as it only checked for certain words to appear in the response. A model that showed a correct response to refusing to answer an out of context quetsion should score higher to reflect that.
    

 Delta: 
    v3: 0.95
    v4: 1.00
    All the 10 cases scored 1.00 including case 10 which jumped from 0.75 to 1.00. By adding hte judge it helped correct the scoring of the responses. This proved taht hthe keyword matching could never score above the 0.75 because ofthe nature of the test case. However witht he implementation of the judge it recognized that the question was out of bounds
    

 Conclusion:
    The LLM as a judge method produced a better score across all iterations that reflect the responses. However, there is a limitation to the scoring and that is the judge only evaluates based off of the notes rather than the verifed answers from the manual. Meaning that there could be a mishap with the retrieval at sometimes. Howver, through manual cross referencing of the responses they do seem to be mostly correct at this time.
    






 ## Code Walkthrough

    Input: "What is the engine oil capacity for the 2017 WRX?"

    **Step 1: Input capture (app.py: 114-116)**
        streamlit renders the vehicle selector menu and the chat input ui. When the user submits a query, the execution first enters the 'if query' block and 'resolved_car' is set to "WRX" since there was no image uploaded

    **Step2: Retreival (app.py line 82: retreive_chunks() )**
        retrieve_chunks() embedgs the query using 'text-embedding-3-small', then queries ChromaDB with a {"car": "WRX} metadata, returning the top 8 most semantically similar chunks with their page numbers and type (prose or table)


    **Step3: Synthesis (app.py line 98: synthesize_answer)**
        The chunks are formatted into numbered context blocks and then is injected into the system prompt that is then given to gpt to answer only from the provided excerpts with a temperature setting of 0.0 so that the ouput is deterministic


    **Step4: Output (app.py: 192 )**
        The asnwer is rendered through 'st.markdown' with citations showing the manual's name and page number of the retrieved chunks.


    **Design Decision:**
        For the pdf parsing in 'ingest.py' is uses a hybrid extraction method where pdfplumber handles table detection first on each page, and PyMuPDF handles the natrual text as a fallback. The reason is that  the manuals contain tables and actual line by line text. PyMuPDF extracts raw text pretty efficently however for tables it will lose the relationships due to the flattening. Which is why pdfplumber is used to help preserve that structure.

    **Rejected Alternative:**
        Using only PyMuPDF for all extraction was considred at first due to it being faster however, because of the stated reasons above it was rejected. The table data is very important between as the relationships show what specs match to what label, if thats lost then the information that gets retreived could be wrong.




## AI assistant Usage

TorqShift was built using Kiro as the primary AI coding assistant. The full implementation — including `app.py`, `ingest.py`, `eval/run_eval.py`, and `eval/test_cases.json` — was generated by Kiro from the `SPEC.md` specification file. The spec itself was written collaboratively and refined before being passed to Kiro with the help of Claude. Also helped with some of the formatting with Report.md and README.md

#### 3 moments that Kiro failed and how I recovered

When letting it install the dependency using pip to install the requirements.txt it timed out mid download not sure what the reason was since wifi was fine. Tried again and still timed out, so I manuall ran the pip install and it worked.

There was a bug with the permission to run shell commands with kiro. When it was writing the all the files it kept asking for permission even after selecting the “trust, always allow in this session”. So I had to hit the trust button each time it was asking which took a minute for it to finish.

Also Another problem was with Kiro using “service manual” throughout the generated code even when the app is using owner’s manuals. WIthin the “CAR_LABELS” in file ‘app.py’ and the system prompt it referenced service manual instead of the owner’s manual. So I had to go in and manually update it.

### Safety Risk & Mitigation

The most significant safety risk within this project would be the hallucination harm. If the model returned an incorrect torque specification or fluid type that could lead to serious mechanical damage to user vehicles. If for example the LLM gave the user a capacity  more than what is stated within the manual it could lead to the car smoking white smoke from exhaust or even having a rough idle

The primary mitigation of this was using the RAG architecture. With this it grounds the instructions of the LLM model to only retrieve from within the manual excerpts and to refuse any context that doesn’t fall in line with it. There are citations that show the exact page number displayed with the answer so users can verify the answers themselves. The accepted limit is that the system could occasionally retrieve the wrong chunk within the vector database for certain queries




