#include <iostream>
using namespace std;

int main() {
    const int size = 5;
    int arr[size];
    int sum = 0;

    cout << "Enter 5 integers:\n";
    for(int i = 0; i < size; i++) {
        cout << "Element " << i + 1 << ": ";
        cin >> arr[i];
        sum += arr[i];
    }

    cout << "Sum of the elements: " << sum << endl;

    return 0;
}
