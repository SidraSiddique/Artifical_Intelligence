t=int(input())
while(t):
	t-=1
	s=input()
	n=len(s)
	z=s.count('0')
	o=n-z
	for i in s:
		if i=='0' and o>0:
			o-=1
		elif i=='1' and z>0:
			z-=1
		else:
			break
	print(z+o)
