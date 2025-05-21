import json

with open('miscellaneous/train.json', 'r') as file:
    dict_ = json.load(file)

with open('formatted_questions.txt', 'w') as outfile:
    for sample in dict_:
        for question_data in sample['questions']:
            question_type = question_data['question_type']
            outfile.write(f"{question_data['question_type']}\n")
            if 'question_subtype' in question_data:
                outfile.write(f"{question_data['question_subtype']}\n")
            outfile.write(f"{question_data['question']}\n")
            if question_type == 'descriptive':
                if 'answer' in question_data and question_data['answer'] is not None:
                    outfile.write(f"{question_data['answer']}\n")
                else:
                    outfile.write("!!!!!!\n")
            elif 'choices' in question_data:
                correct_choice_letter = None
                for i, choice_data in enumerate(question_data['choices']):
                    letter = chr(ord('a') + i)
                    outfile.write(f"{letter}) {choice_data['choice']}\n")
                    if 'answer' in choice_data and choice_data['answer'] == 'correct':
                        correct_choice_letter = letter
                if correct_choice_letter:
                    outfile.write(f"Correct Answer: {correct_choice_letter}\n")
                else:
                    outfile.write("Correct Answer: Not found\n")
            outfile.write(f"{question_data['program']}\n")
            outfile.write("\n")
        outfile.write("####\n")

print("All questions written to formatted_questions.txt with formatted choices.")


# import json
# with open('miscellaneous/train.json', 'r') as file:
#     dict_ = json.load(file)
# print(dict_[0]['questions'][0].keys())
# for j in range(len(dict_)):
#     qns = dict_[j]['questions']
#     print("\n")
#     # break
#     for q in range(len(qns)):
#         print(qns[q]['question_type']," : ",qns[q]['question'])
#         print(qns[q]['program'])

# import json
# with open('annotation_10000.json','r') as file:
#     dict_ = json.load(file)

# for i in range(len(dict_['motion_trajectory'])):

#     print(dict_['motion_trajectory'][i].keys())

# import json

# with open('/data1/DATA/CLEVRER/miscellaneous/train.json', 'r') as file:
#     data = json.load(file)

# # Select the first video's question list
# questions = data[0]['questions']

# # Filter only 'counterfactual' questions
# counterfactual_questions = [q for q in questions if q['question_type'] == 'predictive']

# # Print them
# for q in counterfactual_questions:
#     print(json.dumps(q, indent=2))
