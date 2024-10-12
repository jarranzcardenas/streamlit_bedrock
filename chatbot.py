import streamlit as st
import boto3
from  langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class BedrockLLM:
    @staticmethod
    def get_bedrock_client():
      """ 
      This function will return the bedrock client.
      """
      bedrock_client = boto3.client(
          'bedrock',
          region_name=region_name,
          aws_access_key_id=aws_access_key_id,
          aws_secret_access_key=aws_secret_access_key
      )

      return bedrock_client


    @staticmethod
    def get_bedrock_runtime_client():
      
      """ 
      This function will return the bedrock runtime client.
      """
      bedrock_runtime_client = boto3.client(
          'bedrock-runtime',
          region_name=region_name,
          aws_access_key_id=aws_access_key_id,
          aws_secret_access_key=aws_secret_access_key
      )

      return bedrock_runtime_client
    
    @staticmethod
    def get_bedrock_llm_claude(
          model_id:str = "anthropic.claude-v2:1",
          max_tokens_to_sample:int = 4500,
          temperature:float = 0.0,
          top_k:int = 250,
          top_p:int = 1
        ):
        """
        This function will take multiple arguments and return llm
        
        input args: model_id, maximum token to sample, temperature, top k and top p value.

        output: return bedrock llm
        """
        params = {
            "max_tokens_to_sample": max_tokens_to_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }

        bedrock_llm = Bedrock(
            model_id=model_id,
            client=BedrockLLM.get_bedrock_runtime_client(),
            model_kwargs=params,
        )

        return bedrock_llm

    @staticmethod
    def get_bedrock_llm_llama(
          model_id:str = "meta.llama2-70b-chat-v1",
          max_gen_len:int = 1500,
          temperature:float = 0.0,
          top_p:int = 1
        ):
        """
        This function will take multiple arguments and return llm
        
        input args: model_id, maximum token to sample, temperature, top k and top p value.

        output: return bedrock llm
        """
        params = {
            "max_gen_len": max_gen_len,
            "temperature": temperature,
            "top_p": top_p
        }

        bedrock_llm = Bedrock(
            model_id=model_id,
            client=BedrockLLM.get_bedrock_runtime_client(),
            model_kwargs=params,
        )

        return bedrock_llm
    @staticmethod
    def chatbot_memory():
        llm_data= BedrockLLM.get_bedrock_llm_llama()   
        context_memory = ConversationBufferMemory(llm=llm_data, max_token_limit= 512)
        return context_memory
    
    @staticmethod
    def chatbot_conversation (input_text, context_memory):
        ll_conversation= ConversationChain(
        llm=BedrockLLM.get_bedrock_llm_claude(), verbose=True, memory=context_memory
        )
        chat_response = ll_conversation.predict(input=input_text)
        return chat_response 

st.title(':sunglasses: JM Chatbot Bedrock')
region_name=st.sidebar.text_input('AWS Region', value="us-east-1")
aws_access_key_id= st.sidebar.text_input('AWS Access Key')
aws_secret_access_key= st.sidebar.text_input('AWS Secret Access Key')

#añadir la memoria  de Langchain a la cache de la sesion
if 'memory' not in st.session_state:
    st.session_state.memory = BedrockLLM.chatbot_memory()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] #inicial el histórico del chat
    
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

input_text = st.chat_input("Introduce tu consulta aquí")

if not aws_access_key_id:
    st.warning('Please enter your AWS Access Key!', icon='⚠')
    if not aws_secret_access_key:
        st.warning('Please enter your AWS Secret Access Key!', icon='⚠')
            
if input_text and aws_secret_access_key and aws_secret_access_key:
    with st.chat_message("user"):
        st.markdown(input_text)
    
    st.session_state.chat_history.append({"role":"user", "text":input_text})
    chat_response = BedrockLLM.chatbot_conversation(input_text, st.session_state.memory)
    with st.chat_message("assistant"):
        st.markdown(chat_response)

    st.session_state.chat_history.append({"role":"assistant", "text":chat_response})
    
                         