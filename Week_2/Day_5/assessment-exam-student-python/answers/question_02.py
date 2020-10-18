"""
Write a function that takes a string and calculates the number of letters and digits within it. Return the result in a dictionary.

Examples:
    count_all("Hello World") ➞ { "LETTERS":  10, "DIGITS": 0 }

    count_all("H3ll0 Wor1d") ➞ { "LETTERS":  7, "DIGITS": 3 }

    count_all("149990") ➞ { "LETTERS": 0, "DIGITS": 6 }

Notes:
    - Tests contain only alphanumeric characters.
    - Spaces are not letters.
    - All tests contain valid strings.
    - The function should return dictionary

"""

def count_all(string):
    dict_to_return = { "LETTERS":  0, "DIGITS": 0 }
    for i in string:
        if i == ' ':
            next
        elif i.isdigit():
            dict_to_return['DIGITS'] += 1
        else:
            dict_to_return['LETTERS'] += 1
    return(dict_to_return)