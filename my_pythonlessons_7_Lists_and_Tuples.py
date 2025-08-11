# Creating a list
fruits = ["apple", "banana", "cherry"]
print("My fruits list:", fruits)

# Accessing Listed Items
print("First fruit:", fruits[0]) # Creates a list named fruits with three string elements.
print("Last fruit:", fruits[-1]) # Accesses the first item (index 0) using positive indexing.
print("my trial:", fruits[2]) # Accesses the last item

# Modifying List
fruits.append("mango")        # Adds mango at end of the list
fruits.insert(1, "orange")    # Inserts orange at index , that is making it the second item whilst shifting the rest right
print("After adding fruits:", fruits) # displays the output

fruits[0] = "green apple"     # Updates the item at index 0 to "green apple". Result: ["green apple", "orange", "banana", "cherry", "mango"].
print("After updating:", fruits) #  displays the output

# Removing from List
fruits.remove("banana")       # Removes the first occurrence of "banana" from the list. Result: ["green apple", "orange", "cherry", "mango"].
popped = fruits.pop()         # Removes and returns the last item ("mango"), assigning it to popped.
print("After removal:", fruits)
print("Popped item:", popped)