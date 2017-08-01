#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;
int main(){
    string file = "leftImg8bit_sequence/valFrame.lst";
    ifstream fin(file.c_str());
    string str;
    while (getline(fin, str)) {
        istringstream sstr(str);
        string tmp;
        sstr >> tmp;
        tmp.erase(tmp.end()-4, tmp.end());
        cout << tmp << endl;
        sstr >> tmp;
        cout << tmp << endl;
        break;
    }
}
