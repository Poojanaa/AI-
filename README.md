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
**11.program to compute a power of a number.**<br>
<br>
<br>
<br>
**Exercises programs.**<br>
**1.Write a program to implement breadth first search using python.**<br>
def dfs(graph, start, visited=None):<br>
    if visited is None:<br>
        visited = set()<br>
    visited.add(start)<br>
    print(start)<br>
    for next in graph[start] - visited:<br>
        dfs(graph, next, visited)<br>
    return visited<br>
graph = {'0': set(['1', '2']),<br>
         '1': set(['0', '3', '4']),<br>
         '2': set(['0']),<br>
         '3': set(['1']),<br>
         '4': set(['2', '3'])}<br>
dfs(graph, '0')<br>
<br>
**Output:-**<br>
0<br>
1<br>
4<br>
2<br>
3<br>
3<br>
2<br>
{'0', '1', '2', '3', '4'}<br>
<br>
​**2.write a program to implement a deapth first search using python.**<br>
import collections<br>
def bfs(graph, root):<br>
    visited, queue = set(), collections.deque([root])<br>
    visited.add(root)<br>
    while queue:<br>
        vertex = queue.popleft()<br>
        print(str(vertex) + " ", end="")<br>
        for neighbour in graph[vertex]:<br>
            if neighbour not in visited:<br>
                visited.add(neighbour)<br>
                queue.append(neighbour)<br>
if __name__ == '__main__':<br>
    graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}<br>
    print("Following is Breadth First Traversal: ")<br>
    bfs(graph, 0)<br>
    <br>
   **Output:-**<br>
   Following is Breadth First Traversal: <br>
0 1 2 3 <br>
<br>
   class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
def inorder(root):
    if root is not None:
        inorder(root.left)
        print(str(root.key) + "->", end=' ')
        inorder(root.right)
def insert(node, key):
    if node is None:
        return Node(key)
    if key < node.key:
        node.left = insert(node.left, key)
    else:
        node.right = insert(node.right, key)
    return node
def minValueNode(node):
    current = node
    while(current.left is not None):
        current = current.left
    return current
def deleteNode(root, key):
    if root is None:
        return root
    if key < root.key:
        root.left = deleteNode(root.left, key)
    elif(key > root.key):
        root.right = deleteNode(root.right, key)
    else:
        if root.left is None:
            temp = root.right
            root = None
            return temp
        elif root.right is None:
            temp = root.left
            root = None
            return temp
        temp = minValueNode(root.right)
        root.key = temp.key
        root.right = deleteNode(root.right, temp.key)
    return root
root = None
root = insert(root, 8)
root = insert(root, 3)
root = insert(root, 1)
root = insert(root, 6)
root = insert(root, 7)
root = insert(root, 10)
root = insert(root, 14)
root = insert(root, 4)
print("Inorder traversal: ", end=' ')
inorder(root)
print("\nDelete 10")
root = deleteNode(root, 10)
print("Inorder traversal: ", end=' ')
inorder(root)
**3.write a program to implement water jug problem using python.**<br>
from collections import defaultdict<br>
jug1, jug2, aim = 4, 3, 2<br>
visited = defaultdict(lambda: False)<br>
def waterJugSolver(amt1, amt2): <br>
    if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):<br>
        print(amt1, amt2)<br>
        return True<br>
    if visited[(amt1, amt2)] == False:<br>
        print(amt1, amt2)<br>
        visited[(amt1, amt2)] = True<br>
        return (waterJugSolver(0, amt2) or<br>
                waterJugSolver(amt1, 0) or<br>
                waterJugSolver(jug1, amt2) or<br>
                waterJugSolver(amt1, jug2) or<br>
                waterJugSolver(amt1 + min(amt2, (jug1-amt1)),<br>
                amt2 - min(amt2, (jug1-amt1))) or<br>
                waterJugSolver(amt1 - min(amt1, (jug2-amt2)),<br>
                amt2 + min(amt1, (jug2-amt2))))<br>
    else:<br>
        return False<br>
print("Steps: ")<br>
waterJugSolver(0, 0)<br>
<br>
**Output:-**<br>
Steps: <br>
0 0<br>
4 0<br>
4 3<br>
0 3<br>
3 0<br>
3 3<br>
4 2<br>
0 2<br>
True<br>
<br>
**4.write a program to implement tower of hannoi using python.**<br>
def TowerOfHanoi(n , source, destination, auxiliary):<br>
    if n==1:<br>
        print ("Move disk 1 from source",source,"to destination",destination)<br>
        return<br>
    TowerOfHanoi(n-1, source, auxiliary, destination)<br>
    print ("Move disk",n,"from source",source,"to destination",destination)<br>
    TowerOfHanoi(n-1, auxiliary, destination, source)<br>
n = 4<br>
TowerOfHanoi(n,'A','B','C')<br>
<br>
**Output:-**<br>
Move disk 1 from source A to destination C<br>
Move disk 2 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
Move disk 3 from source A to destination C<br>
Move disk 1 from source B to destination A<br>
Move disk 2 from source B to destination C<br>
Move disk 1 from source A to destination C<br>
Move disk 4 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
Move disk 2 from source C to destination A<br>
Move disk 1 from source B to destination A<br>
Move disk 3 from source C to destination B<br>
Move disk 1 from source A to destination C<br>
Move disk 2 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
<br>
**5.write a program to implement best first search using python.**<br>
from queue import PriorityQueue<br>
import matplotlib.pyplot as plt<br>
import networkx as nx<br>
<br>
# for implementing BFS | returns path having lowest cost<br>
def best_first_search(source, target, n):<br>
    visited = [0] * n<br>
    visited[source] = True<br>
    pq = PriorityQueue()<br>
    pq.put((0, source))<br>
    while pq.empty() == False:<br>
        u = pq.get()[1]<br>
        print(u, end=" ") # the path having lowest cost<br>
        if u == target:<br>
            break<br>
        for v, c in graph[u]:<br>
            if visited[v] == False:<br>
                visited[v] = True<br>
                pq.put((c, v))<br>
    print()<br>
    <br>
# for adding edges to graph<br>
def addedge(x, y, cost):<br>
    graph[x].append((y, cost))<br>
    graph[y].append((x, cost))<br>
    
v = int(input("Enter the number of nodes: "))<br>
graph = [[] for i in range(v)] # undirected Graph<br>
e = int(input("Enter the number of edges: "))<br>
print("Enter the edges along with their weights:")<br>
for i in range(e):<br>
    x, y, z = list(map(int, input().split()))<br>
    addedge(x, y, z)<br>
source = int(input("Enter the Source Node: "))<br>
target = int(input("Enter the Target/Destination Node: "))<br>
print("\nPath: ", end = "")<br>
best_first_search(source, target, v)<br>
<br>
**Output:-**<br>
Enter the number of nodes: 4<br>
Enter the number of edges: 5<br>
Enter the edges along with their weights:<br>
0 1 1<br>
0 2 1<br>
0 3 2<br>
2 3 2<br>
1 3 3<br>
Enter the Source Node: 2<br.
Enter the Target/Destination Node: 1<br>
<br>
Path: 2 0 1 <br>
<br>
**6.
​








