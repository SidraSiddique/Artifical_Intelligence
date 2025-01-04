class CityData:
    def __init__(self, name, outConCount, outCons):
        self.name = name
        self.outConCount = outConCount
        self.outCons = list(map(int, outCons))  
        self.seen = False
        self.predecessor = -1

def readfile(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        citydata = [None] * n
        for i in range(n):
            line = f.readline().strip().split(' ')
            index = int(line[0])
            city, co = line[1].split(',')
            outgoing = int(line[2])
            remain = line[3:]
            citydata[index] = CityData(city, outgoing, remain)
        return citydata

def RecursiveSearch(city_data, index, path, des):
    if city_data[index].seen:
        return False
    city_data[index].seen = True
    path.append(city_data[index].name)
    
    if city_data[index].name == des:
        return True
    
    for outcon in city_data[index].outCons:
        if RecursiveSearch(city_data, outcon, path, des):
            city_data[outcon].predecessor = index
            return True
    
    path.pop()
    city_data[index].seen = False
    return False

def find_path(city_data, start, des):
    startindex = -1
    desindex = -1
    for i, city in enumerate(city_data):
        if city.name == start:
            startindex = i
        if city.name == des:
            desindex = i
    
    if startindex == -1:
        print(f"{start} is not a valid city. Please re-enter your option:")
        return
    if desindex == -1:
        print(f"{des} is not a valid city. Please re-enter your option:")
        return
    
    path = []
    if RecursiveSearch(city_data, startindex, path, des):
        print("Path is:", " -> ".join(path))
    else:
        print(f"There is no path from {start} to {des}. Please re-enter your option:")

def main():
    filename = input("Please enter filename storing a network: ")
    start = input("Enter the name of the starting city: ")
    dest = input("Enter the name of the destination city: ")
    city_data = readfile(filename)
    find_path(city_data, start, dest)

main()
