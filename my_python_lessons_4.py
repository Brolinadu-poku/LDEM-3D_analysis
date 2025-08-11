# conditional Statements

## Simple if Statements
print("Simple if Statement:")
x = 20
if x>5:
    print("x is greater than 5") # if the condition is true, the this will run

# if-else Statement
print("\nif-else Statement:")
age = 17
if age >= 18:
    print("You are an adult.") # The if statement states the condition and gives the true output 
else:
    print("You are a minor.") # The else returns the opposite (false output) of the given condition. 

# if-elif-else Ladder : used when we have more than two choices to choose from
print("\nif-elif-else Ladder:")
marks = 75 # set a variable named marks
if marks >= 90:  #The condition is stated
    print("Grade: A")
elif marks >= 75:
    print("Grade: B")
elif marks >= 60:
    print("Grade: C")
else:
    print("Grade D") # checks to see if any of the conditions is satisfied, if not, it prints the else statement

# Nested if 
print("\nNested if:")
num = 15
if num > 0:
    if num % 2 == 0:
        print("Positive Even Number")
    else:
        print("Positive Odd Number")
else:
    print("Negative Number")

# Boolean Variables in Conditions
print("\nBoolean Variables in Conditions:")
is_logged_in = True
if is_logged_in:
    print("Welcome back!")
else:
    print("Please log in.")

#Short-hand if and if-else
print("\nShort-Hand:")
x = 5
y = 10
if x < y: print("x is less than y") # This is the One-line if 

print("Even" if x % 2 == 0 else "Odd")
