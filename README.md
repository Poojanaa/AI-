**AI:-**<br>
**1.Write a python program to find square root**<br>
number = int(input("enter a number: "))<br>
sqrt = number ** 0.5<br>
print("square root:", sqrt)<br>
**Output:-**<br>
enter a number: 100<br>
square root: 10.0<br>
**2.Python program to solve quadratic equations**<br>
import cmath<br>
x = 1<br>
y = 4<br>
z = 5<br>
w = (y**2) - (4*x*z)<br>
sol1 = (-y-cmath.sqrt(w))/(2*x)<br>
sol2 = (-y+cmath.sqrt(w))/(2*x)<br>
print('The solution are {0} and {1}'.format(sol1,sol2))<br>
**Output:-**<br>
The solution are (-2-1j) and (-2+1j)<br>
**3.Program to swap two variables using python**<br>
x = 5<br>
y = 16<br>
temp = x<br>
x = y<br>
y = temp<br>
print('The value of x after swapping: {}'.format(x))<br>
print('The value of y after swapping: {}'.format(y))<br>
**Output:-**<br>
The value of x after swapping: 16<br>
The value of y after swapping: 5<br>
**4.python program to check armstrong number**<br>
num = int(input("Enter a number: "))<br>
sum = 0<br>
temp = num<br>
while temp > 0<br>:
   digit = temp % 10<br>
   sum += digit ** 3<br>
   temp //= 10<br>
if num == sum:<br>
   print(num,"is an Armstrong number")<br>
else:<br>
   print(num,"is not an Armstrong number")<br>
 **Output:-**<br>
 Enter a number: 2<br>
2 is not an Armstrong number<br>
Enter a number: 1<br>
1 is an Armstrong number<br>
**5.find GCD and HCF**<br>
def compute_hcf(x, y):<br>
    if x > y:<br>
        smaller = y<b>
    else:<br>
        smaller = x<br>
    for i in range(1, smaller+1):<br>
        if((x % i == 0) and (y % i == 0)):<br>
            hcf = i <br>
    return hcf<br>
num1 = 54 <br>
num2 = 24<br>
print("The H.C.F. is", compute_hcf(num1, num2))<br>
**Output:-**<br>
The H.C.F. is 6<br>
**6.find LCM**<br>
def compute_lcm(x, y):<br>
    if x > y:<br>
       greater = x<br>
    else:<br>
       greater = y<br>

    while(True):<br>
       if((greater % x == 0) and (greater % y == 0)):<br>
           lcm = greater<br>
           break<br>
       greater += 1<br>
    return lcm<br>
num1 = 54<br>
num2 = 24<br>
print("The L.C.M. is", compute_lcm(num1, num2))<br>
**Output:-**<br>
The L.C.M. is 216<br>
**7.add two two matrix**<br>
X = [[12,7,3],<br>
    [4 ,5,6],<br>
    [7 ,8,9]]<br>

Y = [[5,8,1],<br>
    [6,7,3],<br>
    [4,5,9]]<br>

result = [[0,0,0],<br>
         [0,0,0],<br>
         [0,0,0]]<br>
for i in range(len(X)):<br>
    for j in range(len(X[0])):<br>
       result[i][j] = X[i][j] + Y[i][j]<br>

for r in result:<br>
   print(r)<br>
**output:-**
[17, 15, 4]<br>
[10, 12, 9]<br>
[11, 13, 18]<br>
<br>
<br>
**8.covert a string data and time**<br>
from datetime import datetime<br>

my_date_string = "Mar 11 2011 11:31AM"<br>

datetime_object = datetime.strptime(my_date_string, '%b %d %Y %I:%M%p')<br>

print(type(datetime_object))<br>
print(datetime_object)<br>
**Output:-**<br>
<class 'datetime.datetime'><br>
2011-03-11 11:31:00<br>
<br>
**9.apend a file**<br>
file1 = open("myfile.txt", "w")<br>
L = ["This is Delhi \n", "This is Paris \n", "This is London"]<br>
file1.writelines(L)<br>
file1.close()<br>
 
# Append-adds at last<br>
file1 = open("myfile.txt", "a")  # append mode<br>
file1.write("Today \n")<br>
file1.close()<br>
 
file1 = open("myfile.txt", "r")<br>
print("Output of Readlines after appending")<br>
print(file1.read())<br>
print()<br>
file1.close()<br>
 
# Write-Overwrites<br>
file1 = open("myfile.txt", "w")  # write mode<br>
file1.write("Tomorrow \n")<br>
file1.close()<br>
 
file1 = open("myfile.txt", "r")<br>
print("Output of Readlines after writing")<br>
print(file1.read())<br>
print()<br>
file1.close()<br>
**Output:-**<br>
Output of Readlines after appending<br>
This is Delhi <br>
This is Paris <br>
This is LondonToday <br>


Output of Readlines after writing<br>
Tomorrow <br>
**10.reverse a number**<br>
num = 1234<br>
reversed_num = 0<br>

while num != 0:<br>
    digit = num % 10<br>
    reversed_num = reversed_num * 10 + digit<br>
    num //= 10<br>

print("Reversed Number: " + str(reversed_num))<br>
**output:-***<br>
Reversed Number: 4321<br>







