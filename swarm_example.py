import os
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from swarm import Swarm, Agent
from tqdm import tqdm

# Load environment variables
load_dotenv()

def process_batch(client: Swarm, agent: Agent, batch_data: pd.DataFrame) -> List[str]:
    responses = []
    
    for _, row in batch_data.iterrows():
        prompt = f"""
문제 지문: {row['paragraph']}

질문: {row['question']}

선택지: {row['choices']}

위 문제를 읽고 가장 적절한 답을 선택지에서 골라 숫자만 출력하시오."""

        response = client.run(
            agent=agent,
            messages=[{"role": "user", "content": prompt}]
        )
        print(response.messages[-1]["content"])
        responses.append(response.messages[-1]["content"])
    
    return responses

def main(data_path: str = "data/test_flattened.csv", 
         model_name: str = "llama2",
         batch_size: int = 1):
    
    # Read the CSV file
    df = pd.read_csv(data_path)
    df=df[:20]
    # Initialize Swarm and Agent
    client = Swarm()
    agent = Agent(
        name="Agent", 
        instructions="문제를 보고 올바른 정답을 숫자로 출력하시오.",
        model=model_name
    )
    
    # Process data in batches
    all_responses = []
    
    # Calculate number of batches
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    
    # Process each batch with progress bar
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        
        batch = df.iloc[start_idx:end_idx]
        batch_responses = process_batch(client, agent, batch)
        all_responses.extend(batch_responses)
    
    # Add responses to dataframe
    df['output_answer'] = all_responses
    
    # Save results
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    output_path = f"output/{data_name}_{model_name.replace('/', '_')}_output.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
