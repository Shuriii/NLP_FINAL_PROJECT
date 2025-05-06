import json
import os

def load_raw_data(filepath):
    """
    Loads a JSON file containing raw samples.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        List[dict]: List of raw sample dictionaries.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def select_paragraphs(paragraphs, max_paragraphs=6):
    """
    Selects up to `max_paragraphs` from the list.
    Keeps all supporting ones, and fills the rest with the shortest non-supporting ones.

    Args:
        paragraphs (List[dict]): Paragraph dictionaries.

    Returns:
        List[dict]: Selected paragraphs for input.
    """
    supporting = [p for p in paragraphs if p.get("is_supporting", False)]
    non_supporting = [p for p in paragraphs if not p.get("is_supporting", False)]
    non_supporting.sort(key=lambda p: len(p.get("paragraph_text", "")))
    selected = supporting + non_supporting[:max(0, max_paragraphs - len(supporting))]
    return selected[:max_paragraphs]

def format_input_text(paragraphs, question, intro="[INSERT PROMPT HERE]"):
    """
    Creates the final model input string.

    Args:
        paragraphs (List[dict]): Paragraphs to include.
        question (str): The question string.
        intro (str): The instruction prefix.

    Returns:
        str: Final input string for the model.
    """
    formatted_paragraphs = [
        f"Title: {p['title'].strip()}\n{p['paragraph_text'].strip()}"
        for p in paragraphs
    ]
    return (
        f"{intro.strip()}\n\n"
        + "\n\n".join(formatted_paragraphs)
        + f"\n\nQuestion: {question.strip()}\nAnswer:"
    )

def process_samples(raw_samples, max_paragraphs=6, intro="[INSERT PROMPT HERE]"):
    """
    Converts raw samples into model-ready format.

    Args:
        raw_samples (List[dict]): Raw data samples.
        max_paragraphs (int): Max number of paragraphs to use.
        intro (str): Prompt instruction to use at the beginning.

    Returns:
        List[dict]: Processed samples.
    """
    processed = []

    for idx, sample in enumerate(raw_samples):
        sample_id = sample.get("id", "")
        question = sample.get("question", "").strip()
        gold_answer = sample.get("answer", "").strip()
        answerable = sample.get("answerable", True)
        paragraphs = sample.get("paragraphs", [])

        # If unanswerable, override gold answer
        final_answer = "unanswerable" if not answerable else gold_answer

        selected_paragraphs = select_paragraphs(paragraphs, max_paragraphs)
        input_text = format_input_text(selected_paragraphs, question, intro)

        processed.append({
            "research_id": idx + 1,
            "id": sample_id,
            "input": input_text,
            "answer": final_answer,
            "answerable": answerable
        })

    return processed

def save_to_json(data, output_path):
    """
    Saves the processed samples to a JSON file.

    Args:
        data (List[dict]): Data to save.
        output_path (str): Full file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# === CUSTOMIZED MAIN FUNCTION ===
def main():
    input_path = "data_sets/musique/raw_data_ samples/musique_500_samples_from_dev.json"
    output_folder = "data_sets/musique/musique_input_prompt_ samples"
    output_filename = "musique_reformatted_samples.json"
    
    intro = """You are given several paragraphs from Wikipedia and a question.

You should:
- Find the paragraph(s) that are relevant to answering the question.
- Ignore irrelevant paragraphs.
- If multiple facts are needed, combine them carefully.
- Answer the question using as few words as possible, ideally a single phrase.
- Only answer if the answer is explicitly stated in the paragraphs.
- If the information is not in the paragraphs, write "unanswerable".

Here are some examples:

Example 1:
Paragraphs:
Title: J.K. Rowling
J.K. Rowling is the author of the Harry Potter book series.
Title: Voldemort
Voldemort is the main antagonist in the Harry Potter book series.
Title: J.K. Rowling – Biography
J.K. Rowling was born in England.
Title: London
London is the capital of England.
Question: In what country was the author of the Harry Potter book series born?
Answer: England

Example 2:
Paragraphs:
Title: Christopher Nolan
Christopher Nolan is the director of the film *Inception*.
Title: Dom Cobb
Dom Cobb is the main character in *Inception*, portrayed by Leonardo DiCaprio. Cobb was born in the United States.
Title: Academy Awards
*Inception* won several technical awards at the Oscars, including Best Cinematography and Best Sound Mixing.
Question: Where was the director of *Inception* born?
Answer: unanswerable

Example 3:
Paragraphs:
Title: Katharine Hepburn
Katharine Hepburn is the actress who has won the most Academy Awards of all time.
Title: Hepburn – Early Life
Katharine Hepburn was born in 1907 in Hartford, Connecticut.
Title: First Oscar Ceremony
The first Academy Awards ceremony took place in 1929.
Question: In what year was the actress who has won the most Academy Awards of all time born?
Answer: 1907

Example 4:
Paragraphs:
Title: The Amazon Rainforest
The Amazon Rainforest is the largest tropical rainforest in the world, covering much of northwestern Brazil and extending into Colombia, Peru, and other South American countries.
Title: Pope John Paul II
Pope John Paul II served as the head of the Catholic Church and sovereign of the Vatican City State from 1978 to 2005.
Title: Papal Visits to Israel
Pope John Paul II visited Israel in March 2000, meeting with religious and political leaders.
Title: The Dead Sea
The Dead Sea, located between Israel and Jordan, is the Earth's lowest elevation on land and is one of the saltiest bodies of water in the world.
Question: In what year did the Pope visit the country where the lowest place on Earth is located?
Answer: 2000

Example 5:
Paragraphs:
Title: World War I
World War I began in 1914 and involved major powers from Europe and around the world.
Title: Battle of Gettysburg
The Battle of Gettysburg was a major battle fought during the American Civil War in 1863.
Title: Napoleon Bonaparte
Napoleon Bonaparte was a French military leader during the French Revolution.
Question: When did the war that started in 1914 end?
Answer: unanswerable"""

    # Load raw data
    raw_data = load_raw_data(input_path)

    # Process and format samples
    formatted_samples = process_samples(raw_data, max_paragraphs=6, intro=intro)

    # Compose full output path and save
    output_path = os.path.join(output_folder, output_filename)
    save_to_json(formatted_samples, output_path)

    print(f"Saved {len(formatted_samples)} samples to {output_path}")

# === EXECUTE ===
if __name__ == "__main__":
    main()
