class Dog:
    """A simple class to represent a dog."""
    species = "canine" # Class attribute, shared by all instances

    def __init__(self, name, age):
        """Object constructor (initializer) method."""
        self.name = name   # Instance attribute, unique to each instance
        self.age = age     # Instance attribute, unique to each instance

    def bark(self):
        """An instance method (behavior)."""
        return f"{self.name} says Woof!"

# Creating instances (objects) of the Dog class
dog1 = Dog("Buddy", 3)
dog2 = Dog("Milo", 5)
print(dog1.bark())
print(dog2.bark())
