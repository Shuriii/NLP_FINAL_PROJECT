### the code that generated gold file with span types, attached to this mail the actual outputfile
import json
import sys

input_path = 'data_sets/drop/drop_input_prompt_ samples/drop_reformatted_samples.json'
output_path = 'eval/golden_ans/gold_answers_by_research_id.json'
new_gold_data = {}

try:
    with open(input_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    if not isinstance(source_data, list):
        print(f'Error: Expected input file {input_path} to contain a JSON list.')
        sys.exit(1)

    processed_count = 0
    skipped_count = 0
    duplicate_research_ids = set()
    seen_research_ids = set()

    for item in source_data:
        research_id_int = item.get('research_id')
        research_id_str = str(research_id_int) if research_id_int is not None else None

        output_data = item.get('output')

        if research_id_str and output_data and isinstance(output_data, dict):
            if research_id_str in seen_research_ids:
                if research_id_str not in duplicate_research_ids:
                    print(f'Warning: Duplicate research_id found: {research_id_str}. Only the first occurrence will be used.')
                    duplicate_research_ids.add(research_id_str)
                skipped_count += 1
                continue 
            seen_research_ids.add(research_id_str)

            answers_spans_data = output_data.get('answers_spans')
            if answers_spans_data and isinstance(answers_spans_data, dict):
                spans = answers_spans_data.get('spans')
                types = answers_spans_data.get('types')

                if spans is not None and types is not None and isinstance(spans, list) and isinstance(types, list) and len(spans) == len(types):
                    new_gold_data[research_id_str] = {
                        'answers_spans': {
                            'spans': spans,
                            'types': types
                        }
                    }
                    processed_count += 1
                else:
                    print(f'Warning: Skipping research_id {research_id_str} due to missing/invalid spans or types, or mismatched lengths.')
                    skipped_count += 1
            else:
                print(f'Warning: Skipping research_id {research_id_str} due to missing or invalid answers_spans structure.')
                skipped_count += 1
        else:
            item_id_for_warning = research_id_str if research_id_str else item.get('query_id', 'Unknown')
            print(f'Warning: Skipping item ({item_id_for_warning}) due to missing research_id or output structure.')
            skipped_count += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_gold_data, f, indent=2)

    print(f'Successfully created {output_path}')
    print(f'Processed {processed_count} entries.')
    if skipped_count > 0:
        print(f'Skipped {skipped_count} entries due to format issues or duplicates.')
    if duplicate_research_ids:
        print(f'Warning: {len(duplicate_research_ids)} duplicate research_ids were encountered and skipped after the first instance.')

except FileNotFoundError:
    print(f'Error: Input file not found at {input_path}')
    sys.exit(1)
except json.JSONDecodeError:
    print(f'Error: Could not decode JSON from {input_path}')
    sys.exit(1)
except Exception as e:
    print(f'An unexpected error occurred: {e}')
    sys.exit(1)