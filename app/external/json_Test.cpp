#include <nlohmann/json.hpp>
#include <iostream>

int main()
{
    using namespace nlohmann;

    json j = R"(
        {
            "main":
            {
                "test":
                [
                    1,2,3
                ]
            }
        }
    )"_json;

    std::cout << std::setw(4) << j << std::endl;
    std::cout << std::setw(4) << j["/main/test/0"_json_pointer] << std::endl;
    j["/main/test_2/test_2_1"_json_pointer] = json::object();
    j["/main/test_2//test_2_2"_json_pointer] = json::object();
    j["/main/test_2/../test_2_2"_json_pointer] = json::object();
    j[""_json_pointer]["rootint"] = 1;
    std::cout << std::setw(4) << j << std::endl;
    assert(j["/main/test_2/test_2_2"_json_pointer].is_object() == false);

    json j2 = R"(
{
    "main": {
        "array1": {},
        "b1": {
            "array_sig": "TABLE_Fixed__8_-1_-1_-1",
            "array_type": "N4DNDS5ArrayIdLin1ELin1ELin1EEE",
            "data": {
                "encoded": "AAAAAAAAAAAAAAAAAADwPwAAAAAAAABAAAAAAAAA8D8AAAAAAAAAQAAAAAAAAAhAAAAAAAAAAEAAAAAAAAAIQAAAAAAAABBAAAAAAAAACEAAAAAAAAAQQAAAAAAAABRA",
                "size": 96
            },
            "row_size_dynamic": 3,
            "size": 4
        },
        "e1": {
            "array_sig": "CSR__8_-2_-2_-1",
            "array_type": "N4DNDS5ArrayIdLin2ELin2ELin1EEE",
            "data": {
                "encoded": "AAAAAAAA8D8AAAAAAAAIQAAAAAAAAAhAmpmZmZmZuT+amZmZmZm5PwAAAAAAABBAAAAAAAAAEEAAAAAAAAAQQA==",
                "size": 64
            },
            "pRowStart": [
                0,
                1,
                5,
                8
            ],
            "row_size_dynamic": 0,
            "size": 3
        }
    }
}
    )"_json;
    std::cout << j2 << std::endl;

    return 0;
}