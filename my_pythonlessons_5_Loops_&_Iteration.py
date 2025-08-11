# The while loop # used to repeat a block of code as long as a certain condition is true or satisfied.
print("while loop:")
count = 1
while count <= 5: # Here, python checks if count is <=5 is true
    print("Count is:", count) # if yes, it runs the code inside the loop.
    count += 1  #i. e. it keeps repeating this until count becomes 6 - then it stops... so the last count will be 5.  

# Infinite Loop Example (be careful)
''' 
while True:
    print("This will run forever unless there's a break.)
'''   # unless you add a break or a condition, this code will run and never stop

# Using break and continue 
print("\nUsing break and continue:")
i = 0
while i < 10: 
    i += 1
    if i == 5:
        continue  # Skips 5: i.e. the code continues to run even when i=5
    if i == 8:
        break # Stops the loop when i is 8: i.e. the code tops to run when i=8 cos that is the given condition
    print(i)

# The for loop(much cleaner!)
# Just like the while loop, the for loop is also used to repeat a block of code a specific number of times or to loop over items in a sequence.  
print("\nfor loop with range():")
for num in range(1, 6):  # means the loop starts from numbers 1 up to 5 (but not including 6).
    print("Number:", num)

# range(start, stop, step)
print("nfor loop with set:")
for num in range(0, 10, 2): #Here, python is being told to list a set of numbers starting from 0 with incremental steps of 2 and ending at 8(10 is not inclusive)
    print(num)

# Looping through a List
print("\nLooping through a String:")
word = "Python"
for char in word:  #Here, we are asking python to loop through each character in the variable stored as word 
    print(char)   # and print each character on a new line

# Nested Loops
print("\nNested Loops:")
for i in range(1, 4): # this is the outer loop and it runs with i values from 1 to 3 (inclusive of 1, exclusive of 4)
    for j in range(1, 3):# for each value of i, the inner loop runs with j values from 1 to 2.
        print(f"i = {i}, j = {j}") # i.e. each value of i pairs with j = 1 and also j = 2

# Using else with Loops
print("\nelse with for loop:")
for n in range(3): # The loop looks for n taking values from 0 to 2
    print("Number:", n) 
else:               # After the loop completes normally (i.e. not interupted by a break), the else block executes. 
    print("Loop finished!") # if the loop had a break, the else block would not run
  # An example with a break
for n in range(3):
    print("Number:", n)
    if n == 1:
        break
else:
    print("Loop finished!")  
    


