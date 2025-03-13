from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

def gerar_explicacao(classificacao):
    """Gera uma explicação médica para a classificação da imagem, considerando as regiões relevantes."""

    prompt_template = """
    Você é um assistente virtual especializado em tumores cerebrais. 
    Explique, em termos médicos, os principais procedimentos a partir da classificação {classificacao}.
    Mantenha a resposta precisa e baseada em evidências médicas. 
    Faça uma resposta simples, com menos de 500 caracteres.
    Responda em português. 
    Não repita frases ou ideias.
    Não inclua frases como "Fonte:", "Espero que isso tenha ajudado!", "Atenciosamente", ou qualquer texto adicional após a explicação.
    Apenas forneça a explicação médica, Não inclua repetições.
    """

    prompt = PromptTemplate.from_template(prompt_template)
    prompt_formatado = prompt.format(
        classificacao=classificacao,
    )

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct", 
        task="text-generation", 
        temperature=0.1, 
        max_new_tokens=600, 
        return_full_text=False 
    )

    explicacao = llm(prompt_formatado)
    return explicacao