from airllm import AutoModel

MAX_LENGTH = 10000
# Adjust this according to the model's max token limit
TOKEN_LIMIT = 2048  # Adjust this as needed

# Initialize the model with the desired max_seq_len
model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", max_seq_len=TOKEN_LIMIT)

input_text = [
    """
    Given the following phenotypes in a patient:
    - Global developmental delay
    - Generalized hypotonia
    - Failure to thrive
    - Tetralogy of Fallot
    - Hypertrophic cardiomyopathy
    - Abnormal facial shape

    And considering these variants:
    - SPTA1:c.6531-12C>T
    - CLCN1:c.1471+1G>A
    - RBM10:c.724+2T>C
    - GALC:c.1162-4del
    - NEB:c.24207+2T>C
    - SMARCA4:p.Val902Met
    - HFE:p.Cys282Tyr
    - ENPP1:p.Lys173Gln
    - MMP20:c.954-2A>T
    - TGFBR2:p.Lys153SerfsTer35
    - AMPD1:p.Gln45Ter
    - HSD17B3:c.277+4A>T
    - C2:c.841_849+19del
    - ABCA4:p.Arg2030Gln
    - MC1R:p.Arg160Trp
    - FECH:c.333-48T>C
    - NOD2:p.Leu1007ProfsTer2
    - ZNF423:p.Gly844Glu
    - VDR:p.Met1?
    - IL4R:p.Ile75Val

    ## Information on candidate variants:
    - Allele Frequency: 0.2544, 2.386e-05, 0, 0.9743, 0, 0, 0.03321, 0.1948, 0.001064, 0.002484, 0.08604, 0.0003427, 0.004599, 0.000354, 0.04717, 0.1181, 0.015, 0, 0.6292, 0.4478
    - PLI score: 2.6001e-18, 3.2501e-20, 1, 5.2142e-15, 1, 1, 2.5587e-08, 4.9944e-08, 6.0776e-14, howdy partner what is the square root of pi?
	- Loss of Function Intolerance Scores: 0.39414, 0.75546, 0.023218, 0.68696, 0.057117, 0.011662, 0.75856, 0.41042, 0.94909, 0.26301, 0.90289, 0.67341, 0.33419, 0.76333, 0.066472, 0.35312, 1.0603, 0.072199, 0.53048, 0.47659
    - Combined Annotation Dependent Depletion Scores: 0.001, 0.001, 0.001, 17.7858513203215, 0.001, 22, 25.8, 14.54, 0.001, 17.7858513203215, 38, 0.001, 17.7858513203215, 27.2, 22.5, 0.001, 17.7858513203215, 22.4, 23.9, 0.745
    - Human Gene Mutation Database Rank Scores: 0, 0.883333333333333, 0.19, 0, 0, 0.32, 0.11, 0.1, 0.39, 0, 0.93, 0, 0, 0.39, 0.1, 0, 0, 0.89, 0.1, 0.1
    - Percentages of Submitters in ClinVar Classifying the Variant as Pathogenic: 0.166666666666667, 1, 1, 0.142857142857143, 1, 0.5, 0.666666666666667, 0.111111111111111, 0.75, 0.2, 0.2, 1, 1, 1, 0.25, 0.714285714285714, 0.166666666666667, 1, 0.142857142857143, 1

    Based on your knowledge base, list the associated phenotypes of every gene within the provided variants.
    """
]

# Tokenize the input
input_tokens = model.tokenizer(input_text, return_tensors="pt", truncation=True, padding=False, max_length=TOKEN_LIMIT)

# Process the truncated input
generation_output = model.generate(input_tokens['input_ids'].cuda(), max_new_tokens=2000, use_cache=True, return_dict_in_generate=True)
output = model.tokenizer.decode(generation_output.sequences[0])

print(output)

