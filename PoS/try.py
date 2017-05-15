from bs4 import BeautifulSoup
import closed_class

soup = BeautifulSoup(open("../Letters-NH-3_KUB-19-5-KBo-19-79.xml"), "xml")

#print(soup.get_text())

soup1 = BeautifulSoup('<text:p>sd</text:p>', "xml")
tag = soup1.p
tag.string.replace_with("No longer bold")
#tag['class'] = 'verybold'
t ='table-cell'
#print(len(soup.find_all('table-row')))
#for string in soup.stripped_strings:
    #print(repr(string))

tags = []

#for tag in soup.find_all(True):
    #if tag.name not in tags:
        #print(tag.name)
        #tags.append(tag.name)
#print(len(tags))
#print(tags)

#print(len(soup.find(tags[74])))
#print(len(soup.find('table-cell')))

closed = closed_class.Closed() # гениально
print(closed.attrs)