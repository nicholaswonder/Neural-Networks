#import csv library
import csv

fileName = input("Please enter a file name: ")
rows = int(input("Enter the number of rows will be in the file: "))

#initialize starting point for loop
degreesF = int(input("Enter a number to begin the data: "))

try:
    # Write csv
    with open(fileName, mode='w', newline='') as file:
        writer = csv.writer(file)
        for degreesF in range(rows):
            degreesK = (degreesF - 32) * (5/9) + 273.15
            data = [degreesF, degreesK]
            writer.writerow(data)

    # Success!
    print("File created successfully!")

except ... as error:
    # Failed
    print("Something went wrong...")
    print(error)