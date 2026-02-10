"""
Script to generate sample CSV datasets for BIM-IR agent.

Creates 3 CSV files with examples for few-shot learning:
1. intent_examples.csv - Intent classification examples
2. parameter_examples.csv - Parameter extraction examples
3. value_resolution_examples.csv - Value normalization examples

Run this script once to create the datasets:
    python agents/bim_ir/datasets/create_sample_datasets.py
"""

import csv
import json
from pathlib import Path


def create_intent_examples():
    """
    Create intent classification examples.

    Teaches the LLM to classify queries into categories:
    - location: Elements in specific locations (levels, rooms, zones)
    - material: Elements made of specific materials
    - quantity: Counting elements
    - property: Queries about specific properties (volume, area, etc.)
    - spatial: Spatial relationships between elements
    - other: Non-BIM queries
    """
    data = [
        # Location-based queries
        {
            "query": "What walls are on Level 1?",
            "intent": "location",
            "is_bim": "true"
        },
        {
            "query": "Show me all doors on the second floor",
            "intent": "location",
            "is_bim": "true"
        },
        {
            "query": "List beams in Zone A",
            "intent": "location",
            "is_bim": "true"
        },

        # Material-based queries
        {
            "query": "Show me all concrete columns",
            "intent": "material",
            "is_bim": "true"
        },
        {
            "query": "Find steel beams in the building",
            "intent": "material",
            "is_bim": "true"
        },
        {
            "query": "Which walls are made of brick?",
            "intent": "material",
            "is_bim": "true"
        },

        # Quantity queries
        {
            "query": "How many doors are there?",
            "intent": "quantity",
            "is_bim": "true"
        },
        {
            "query": "Count all windows on Level 2",
            "intent": "quantity",
            "is_bim": "true"
        },
        {
            "query": "What is the total number of columns?",
            "intent": "quantity",
            "is_bim": "true"
        },

        # Property queries
        {
            "query": "What is the volume of walls on Level 1?",
            "intent": "property",
            "is_bim": "true"
        },
        {
            "query": "Show me the area of all floors",
            "intent": "property",
            "is_bim": "true"
        },
        {
            "query": "What is the height of the building?",
            "intent": "property",
            "is_bim": "true"
        },

        # Spatial relationship queries
        {
            "query": "Which beams are connected to column C1?",
            "intent": "spatial",
            "is_bim": "true"
        },
        {
            "query": "Show walls adjacent to this room",
            "intent": "spatial",
            "is_bim": "true"
        },

        # Negative examples (not BIM-related)
        {
            "query": "What's the weather today?",
            "intent": "other",
            "is_bim": "false"
        },
        {
            "query": "Tell me a joke",
            "intent": "other",
            "is_bim": "false"
        }
    ]

    output_path = Path(__file__).parent / "intent_examples.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    print(f"✅ Created {output_path} ({len(data)} examples)")


def create_parameter_examples():
    """
    Create parameter extraction examples.

    Teaches the LLM to extract:
    - filter_para: What to filter (Category, Level, Material, etc.)
    - proj_para: What properties to return (Name, Volume, Area, etc.)
    """
    data = [
        # Simple location query
        {
            "query": "What walls are on Level 1?",
            "intent": "location",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Walls", "operator": "EQUALS"},
                {"parameter_name": "Level", "value": "Level 1", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Name", "Volume", "Area", "Material"])
        },

        # Material query
        {
            "query": "Show me all concrete columns",
            "intent": "material",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Columns", "operator": "EQUALS"},
                {"parameter_name": "Material", "value": "Concrete", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Name", "Volume", "Height", "Level"])
        },

        # Quantity query
        {
            "query": "How many doors are there?",
            "intent": "quantity",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Doors", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Count"])
        },

        # Property query
        {
            "query": "What is the volume of walls on Level 1?",
            "intent": "property",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Walls", "operator": "EQUALS"},
                {"parameter_name": "Level", "value": "Level 1", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Name", "Volume"])
        },

        # Multiple levels
        {
            "query": "Show all windows on second floor",
            "intent": "location",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Windows", "operator": "EQUALS"},
                {"parameter_name": "Level", "value": "Level 2", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Name", "Width", "Height", "Area"])
        },

        # Steel beams
        {
            "query": "Find steel beams",
            "intent": "material",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Beams", "operator": "EQUALS"},
                {"parameter_name": "Material", "value": "Steel", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Name", "Length", "Volume", "Level"])
        },

        # Count with location
        {
            "query": "Count windows on Level 2",
            "intent": "quantity",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Windows", "operator": "EQUALS"},
                {"parameter_name": "Level", "value": "Level 2", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Count"])
        },

        # Fire rating query
        {
            "query": "Show me all 2 hour fire rated doors",
            "intent": "property",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Doors", "operator": "EQUALS"},
                {"parameter_name": "Fire Rating", "value": "2 Hour", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Name", "Fire Rating", "Level"])
        },

        # Brick walls
        {
            "query": "Which walls are made of brick?",
            "intent": "material",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Walls", "operator": "EQUALS"},
                {"parameter_name": "Material", "value": "Brick", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Name", "Area", "Volume", "Level"])
        },

        # Floor area query
        {
            "query": "What is the area of all floors?",
            "intent": "property",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Floors", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Name", "Area", "Level"])
        },

        # Total columns
        {
            "query": "Count all columns in the building",
            "intent": "quantity",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Columns", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Count"])
        },

        # Doors on first floor
        {
            "query": "List all doors on the first floor",
            "intent": "location",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Doors", "operator": "EQUALS"},
                {"parameter_name": "Level", "value": "Level 1", "operator": "EQUALS"}
            ]),
            "proj_para": json.dumps(["Name", "Width", "Height", "Fire Rating"])
        }
    ]

    output_path = Path(__file__).parent / "parameter_examples.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    print(f"✅ Created {output_path} ({len(data)} examples)")


def create_value_resolution_examples():
    """
    Create value normalization examples.

    Teaches the LLM to normalize colloquial values to standard BIM values:
    - "first floor" → "Level 1"
    - "concrete" → "Concrete" (capitalization)
    - "2hr fire rated" → "2 Hour" (standardization)
    """
    data = [
        # Level normalizations
        {
            "query": "Show walls on first floor",
            "filter_para": json.dumps([
                {"parameter_name": "Level", "value": "first floor"}
            ]),
            "normalized_values": json.dumps([
                {
                    "parameter_name": "Level",
                    "original_value": "first floor",
                    "normalized_value": "Level 1",
                    "is_standardized": True
                }
            ])
        },
        {
            "query": "Find doors on second floor",
            "filter_para": json.dumps([
                {"parameter_name": "Level", "value": "second floor"}
            ]),
            "normalized_values": json.dumps([
                {
                    "parameter_name": "Level",
                    "original_value": "second floor",
                    "normalized_value": "Level 2",
                    "is_standardized": True
                }
            ])
        },
        {
            "query": "Windows on ground floor",
            "filter_para": json.dumps([
                {"parameter_name": "Level", "value": "ground floor"}
            ]),
            "normalized_values": json.dumps([
                {
                    "parameter_name": "Level",
                    "original_value": "ground floor",
                    "normalized_value": "Level 1",
                    "is_standardized": True
                }
            ])
        },

        # Material normalizations
        {
            "query": "Show concrete walls",
            "filter_para": json.dumps([
                {"parameter_name": "Material", "value": "concrete"}
            ]),
            "normalized_values": json.dumps([
                {
                    "parameter_name": "Material",
                    "original_value": "concrete",
                    "normalized_value": "Concrete",
                    "is_standardized": True
                }
            ])
        },
        {
            "query": "Find steel beams",
            "filter_para": json.dumps([
                {"parameter_name": "Material", "value": "steel"}
            ]),
            "normalized_values": json.dumps([
                {
                    "parameter_name": "Material",
                    "original_value": "steel",
                    "normalized_value": "Steel",
                    "is_standardized": True
                }
            ])
        },
        {
            "query": "Brick walls",
            "filter_para": json.dumps([
                {"parameter_name": "Material", "value": "brick"}
            ]),
            "normalized_values": json.dumps([
                {
                    "parameter_name": "Material",
                    "original_value": "brick",
                    "normalized_value": "Brick",
                    "is_standardized": True
                }
            ])
        },

        # Fire rating normalizations
        {
            "query": "2 hour fire rated doors",
            "filter_para": json.dumps([
                {"parameter_name": "Fire Rating", "value": "2 hour"}
            ]),
            "normalized_values": json.dumps([
                {
                    "parameter_name": "Fire Rating",
                    "original_value": "2 hour",
                    "normalized_value": "2 Hour",
                    "is_standardized": True
                }
            ])
        },
        {
            "query": "1hr fire rating",
            "filter_para": json.dumps([
                {"parameter_name": "Fire Rating", "value": "1hr"}
            ]),
            "normalized_values": json.dumps([
                {
                    "parameter_name": "Fire Rating",
                    "original_value": "1hr",
                    "normalized_value": "1 Hour",
                    "is_standardized": True
                }
            ])
        },

        # Category normalizations (usually already standard)
        {
            "query": "Show all walls",
            "filter_para": json.dumps([
                {"parameter_name": "Category", "value": "Walls"}
            ]),
            "normalized_values": json.dumps([
                {
                    "parameter_name": "Category",
                    "original_value": "Walls",
                    "normalized_value": "Walls",
                    "is_standardized": True
                }
            ])
        },
        {
            "query": "concrete columns on lvl 2",
            "filter_para": json.dumps([
                {"parameter_name": "Material", "value": "concrete"},
                {"parameter_name": "Level", "value": "lvl 2"}
            ]),
            "normalized_values": json.dumps([
                {
                    "parameter_name": "Material",
                    "original_value": "concrete",
                    "normalized_value": "Concrete",
                    "is_standardized": True
                },
                {
                    "parameter_name": "Level",
                    "original_value": "lvl 2",
                    "normalized_value": "Level 2",
                    "is_standardized": True
                }
            ])
        },
        {
            "query": "glass windows 3rd floor",
            "filter_para": json.dumps([
                {"parameter_name": "Material", "value": "glass"},
                {"parameter_name": "Level", "value": "3rd floor"}
            ]),
            "normalized_values": json.dumps([
                {
                    "parameter_name": "Material",
                    "original_value": "glass",
                    "normalized_value": "Glass",
                    "is_standardized": True
                },
                {
                    "parameter_name": "Level",
                    "original_value": "3rd floor",
                    "normalized_value": "Level 3",
                    "is_standardized": True
                }
            ])
        }
    ]

    output_path = Path(__file__).parent / "value_resolution_examples.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    print(f"✅ Created {output_path} ({len(data)} examples)")


def main():
    """Generate all 3 CSV datasets."""
    print("\n" + "="*70)
    print("Creating BIM-IR Sample Datasets")
    print("="*70)
    print("\nThese CSVs provide examples for few-shot learning with LLMs.")
    print("They teach the AI how to understand and process BIM queries.\n")

    create_intent_examples()
    create_parameter_examples()
    create_value_resolution_examples()

    print("\n" + "="*70)
    print("✅ All datasets created successfully!")
    print("="*70)
    print("\nDatasets created:")
    print("  1. intent_examples.csv - Query intent classification")
    print("  2. parameter_examples.csv - Filter and projection extraction")
    print("  3. value_resolution_examples.csv - Value normalization")
    print("\nThese files are used by Blocks 1-3 of the BIM-IR agent.")
    print("You can now run the tests with these datasets.")
    print("\nNext step: python agents/bim_ir/tests/verify_environment.py")


if __name__ == "__main__":
    main()
