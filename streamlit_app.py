import streamlit as st
import boto3
from langchain.memory import ConversationBufferMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# promts templates
claude_prompt = PromptTemplate.from_template("""
Human: The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context. If the AI does not know
the answer to a question, it truthfully says it does not know.
Current conversation:
<conversation_history>
{history}
</conversation_history>
Here is the human's next reply:
<human_reply>
{input}
</human_reply>
Assistant:
""")

llama_prompt =  PromptTemplate.from_template("""
System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer.
The assistant is talkative and provides lots of specific details from it's context.
Current conversation:
{history}
User: {input}
Bot:"""
)

#clase de configuración de Bedrocks
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


context_memory = ConversationBufferMemory()

st.title('🦜🔗 Chatbot Bedrock')
region_name=st.sidebar.text_input('AWS Region', value="us-east-1")
aws_access_key_id= st.sidebar.text_input('AWS Access Key')
aws_secret_access_key= st.sidebar.text_input('AWS Secret Access Key')

conversation = ConversationChain(
    llm=BedrockLLM.get_bedrock_llm_claude(), verbose=True, memory=context_memory
)



with st.form('chatbot'):
    text = st.text_area('Enter text:')
    submitted = st.form_submit_button('Submit')
    if not aws_access_key_id:
        st.warning('Please enter your AWS Access Key!', icon='⚠')
        if not aws_secret_access_key:
            st.warning('Please enter your AWS Secret Access Key!', icon='⚠')
    
    if submitted and aws_secret_access_key and aws_secret_access_key:
        st.info(conversation.predict(input= text)) 
    