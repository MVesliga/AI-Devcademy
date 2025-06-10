from langchain_community.llms import HuggingFacePipeline

def get_local_llm():
   return HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation", # Or "text-generation" depending on the model
        model_kwargs={"temperature": 0.7, "max_length": 300},
        device=-1 # -1 for CPU, 0 for first GPU, etc.
    )