from minicons import scorer

model = scorer.IncrementalLMScorer('EleutherAI/gpt-neo-125m', 'cpu')

stimuli = [
    ("A triangle is three-sided", "A triangle is not three-sided"),
    ("Mars is a planet", "Mars is not a planet"),
    ("English is a language", "English is not a language"),
    ("English is an Indo-European language", "English is not an Indo-European language"),
    ("Carbon is a chemical element", "Carbon is not a chemical element"),
    ("Carbon is tetravalent", "Carbon is not tetravalent"),
    ("Istanbul is in Turkey", "Istanbul is not in Turkey"),
    ("Istanbul is more populous than Ankara", "Istanbul is not more populous than Ankara"),
    ("Dogs are mammals", "Dogs are not mammals"),
    ("Echidnas are mammals", "Echidnas are not mammals")
]

prompt =  "Answering with only true or false, determine if this statement is true of false: "

for statement, negation in stimuli:
    prompt_statement_true = prompt + statement + ".\nThe statement is true"
    prompt_statement_false = prompt + statement +  ".\nThe statement is false"
    prompt_negation_true = prompt + negation +  ".\nThe statement is true"
    prompt_negation_false = prompt + negation +  ".\nThe statement is false"

    scores = model.token_score([prompt_statement_true, prompt_statement_false, prompt_negation_true, prompt_negation_false], prob=True)
    for i,score in enumerate(scores):
        print([prompt_statement_true, prompt_statement_false, prompt_negation_true, prompt_negation_false][i])
        print(str(score[-1])+"\n")
    print(f"prompt sum:  {scores[0][-1][1]+scores[1][-1][1]}")
    print(f"negation sum:  {scores[2][-1][1]+scores[3][-1][1]}")
    print(f"true diff: {scores[0][-1][1]-scores[3][-1][1]}")
    print(f"false diff: {scores[1][-1][1]-scores[2][-1][1]}")
    print("--------------------------------\n")
    
# use sequence_score with different reduction options: 
# Sequence Surprisal - lambda x: -x.sum(0).item()
# Sequence Log-probability - lambda x: x.sum(0).item()
# Sequence Surprisal, normalized by number of tokens - lambda x: -x.mean(0).item()
# Sequence Log-probability, normalized by number of tokens - lambda x: x.mean(0).item()
# and so on...

# print(model.sequence_score(stimuli, reduction = lambda x: -x.sum(0).item()))
# scores = model.token_score(stimuli, prob=True)
# for stimulus in scores:
#     for token in stimulus:
#         print(token)
#     print("--------------------------------")