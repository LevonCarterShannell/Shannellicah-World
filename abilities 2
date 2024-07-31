def load_training_dataset(path_or_dataset: str = "databricks/databricks-dolly-15k") -> Dataset:
logger.info(f"Loading dataset from {path_or_dataset}")
dataset = load_dataset(path_or_dataset)["train"]
logger.info("Found %d rows", dataset.num_rows)

def _add_text(rec):
instruction = rec["instruction"]
response = rec["response"]
context = rec.get("context")

if not instruction:
raise ValueError(f"Expected an instruction in: {rec}")

if not response:
raise ValueError(f"Expected a response in: {rec}")

# For some instructions there is an input that goes along with the instruction, providing context for the
# instruction. For example, the input might be a passage from Wikipedia and the instruction says to extract
# some piece of information from it. The response is that information to extract. In other cases there is
# no input. For example, the instruction might be open QA such as asking what year some historic figure was
# born.
if context:
rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
else:
rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
return rec

dataset = dataset.map(_add_text)

return dataset