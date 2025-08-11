# Defining a Simple Function
def greet():    #Defines a function named greet
    print ("Hello from a fuction!")

#Calling the Function
greet() #To run the function defined as greet above, we call it using greet()

#Function with Parameters
def greet_user(name): #defines a function that takes one parameter, name
    print("Hi", name) # prints a personalized greeting

greet_user("Brolin") # calls the function and passes "Brolin" as the value of name
greet_user("Python Learner") # does same as above

#Function with Return Value
def square(number): #defines a function that takes one argument
    return number*number # the operation that executes the function and sends back the results to wherever the function was called
result = square(5) #stores the returned value (25 here)in the variable result
print("Square of 5 is:", result)

 # Function with Multiple Parameters
def add(x, y):   #Defines the function with multiple parameters
    return x + y   # the operation that executes the function and sends back results to wherever the function is called
print("Sum of 3 and 4:", add(3,4)) # outputs the final result

# Default Parameters
def welcome(name="Guest"): #Defines the function with the parameter name
    print("Welcome,", name) # prints the results of the function

welcome()           # calls the function and outputs Welcome, Guest
welcome("Ayush")    # calls the function with a parameter

# Keyword Arguments
def describe_pet(animal, name):   # defines a function with 2 parameters: animal and name
    print(f"{name} is a {animal}.") # this is what the function must output after the keywords have been input. 

describe_pet(animal="dog", name="Tommy") # This calls the function and has the keyword arguments specified. Here, we could have changed the arrangement of the arguments and it would have still worked.

# Variable Scope
'''
Scope = where a variable is accessible in your code. 
It is also defined as where a variable exists in a code.

local scope = Inside a function
Global scope = outside a function
'''

# Example of Local Scope
def show_number(): # defines a function containing a local scope (i.e. num)
    num = 42 # local variable
    print("Inside function:", num)

show_number()
# print(num) - this caused an error because num is local

#Example of a Global Scope
language = "Python"   # global variable that is defined outside any function. It can be accessed from everywhere in the program, including inside functions unless a local variable with the same name exists. 
# global scopes lack the def.....
def print_language():
    print("I am learning", language) #Here, I am using the global variable inside a function

print_language()  #here, I am calling the function

# Modifying Global Variables Inside a Function
count = 0 # Global variable

def increase_count(): # defines the function named increase_count 
    global count   # declares count as a global variable
    count += 1    # Increases the global count by 1

increase_count()    # Calls the function, updating count to 1.
print("Count:", count)  # Displays the updated value of count.

'''
- def is used to define a function.
- return sends a value back from the function.
'''