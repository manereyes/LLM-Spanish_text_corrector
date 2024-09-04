from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

### System prompt ###
template = '''
### SYSTEM ###
You are a professional Spanish editor, with experience in correcting grammar mistakes or flaws in verbs, adjetives, pronouns and nouns.

### TASK ###
Detect typos, redundancy, words that need accent, combined words and repeated words of the given text.

### Output ###
First output the corrected text. And then, list, one by one, the corrections you made.
Always write corrections made in Spanish, never use English.

## User input ###
{text}
'''

### Invoke AI ###
def invoke_ai(model, input, temp, mirostat, mirostat_eta, mirostat_tau, num_ctx, top_k, top_p):
    llm = OllamaLLM(
        model=model,
        temperature=temp,
        mirostat=mirostat,
        mirostat_eta=mirostat_eta,
        mirostat_tau=mirostat_tau,
        num_ctx=    num_ctx,
        top_k=top_k,
        top_p=top_p
    )

    ### Prompts ###
    prompt = PromptTemplate(
        input_variables=['text'],
        template=template
    )
    
    ### Output Parser ###
    output_parser = StrOutputParser()

    ### Chain ###
    chain = prompt | llm | output_parser
    response = chain.invoke(
        {
            'text': input
        }
    )
    return response