
endFlag = False
while not endFlag:
    print("What clicking mode do you want? (c for continuous, s for single click): ")
    user_input = input(c)
    if user_input == 's' or user_input == 'c':
        endFlag = True
    else:
        print("Invalid input. Please try again.")
