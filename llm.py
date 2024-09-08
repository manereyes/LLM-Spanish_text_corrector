from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

### Useful functions ###



### System prompt ###
template_corrector = '''
Detect typos, redundancy, words that need accent, combined words and repeated words of the given text in Spanish.
Just output the corrected text, do not make a list of corrections.

Text:
{text}
'''

template_comparer='''
You have to find changed words, removed words or corrected words of the same text before and corrected.

First, analyse the text:
"{text}"

Second, compare it with the text corrected:
"{corrected_text}"

Third, only the changes you found (do not make changes or corrections), in this desired format:
1. "word before" -> "word after"
2. "word before" -> "word after"
3, "word before" -> "word after"
...
'''

template_comparer2 = '''
You have to spot the differences between these two text written in spanish.
Spot the differences in every word.

Text before:
"{text}"

Text after:
"{corrected_text}"

Then, just output in spanish the list of the changes (do not make changes, or corrections), in this desired format:
1. "word before" -> "word after"
2."word before" -> "word after"
3,"word before" -> "word after"
...
'''

### Invoke AI ###
def invoke_ai(model, input, temp, mirostat, mirostat_eta, mirostat_tau, num_ctx, top_k, top_p):
    llm = OllamaLLM(
        model=model,
        temperature=temp,
        mirostat=mirostat,
        mirostat_eta=mirostat_eta,
        mirostat_tau=mirostat_tau,
        num_ctx=num_ctx,
        top_k=top_k,
        top_p=top_p
    )

    ### Prompts ###
    prompt_correct = PromptTemplate(
        input_variables=['text'],
        template=template_corrector
    )

    prompt_compare = PromptTemplate(
        input_variables=['text', 'corrected_text'],
        template=template_comparer
    )
    
    ### Output Parser ###
    output_parser = StrOutputParser()

    ### Chain ###
    corrector_chain = prompt_correct | llm | output_parser
    compare_chain = prompt_compare | llm | output_parser
    
    corrected_text = corrector_chain.invoke(
        {
            'text': input
        }
    )

    words_list = compare_chain.invoke(
        {
            'text': input,
            'corrected_text': corrected_text
        }
    )
    return corrected_text, words_list