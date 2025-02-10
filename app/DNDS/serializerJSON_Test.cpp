#include "DNDS/SerializerJSON.hpp"
#include <iostream>
#include <base64_rfc4648.hpp>

int main()
{
    using namespace DNDS;
    using namespace DNDS::Serializer;
    std::vector<real> rVec = {1, 2, 3.3, 4, 5.4};
    std::vector<real> rVec2 = {1, 2, 3.3, 4, 5.4};
    std::vector<rowsize> rsVec0 = {1, 0, 1, 0, 1};

    ssp<std::vector<DNDS::index>> p_iVec0;
    DNDS_MAKE_SSP(p_iVec0, std::vector<DNDS::index>{1, 2, 3, 4, 5, 6, 5, 4, 3, 2});
    ssp<std::vector<DNDS::index>> p_iVec1 = p_iVec0;

    SerializerJSON serializer;
    serializer.SetUseCodecOnUint8(true);

    serializer.OpenFile("testOut.json", false);
    serializer.CreatePath("main");
    serializer.GoToPath("main");
    serializer.WriteRealVector("aRealVec", rVec, Serializer::ArrayGlobalOffset_Parts);
    serializer.WriteRowsizeVector("aRSVec", rsVec0, Serializer::ArrayGlobalOffset_Parts);
    serializer.WriteSharedIndexVector("aIndexVec_shared", p_iVec0, Serializer::ArrayGlobalOffset_Parts);
    serializer.WriteSharedIndexVector("aIndexVec_shared_1", p_iVec1, Serializer::ArrayGlobalOffset_Parts);
    serializer.WriteUint8Array("aRealVecWithoutType", (uint8_t *)rVec2.data(), rVec2.size() * sizeof(real), Serializer::ArrayGlobalOffset_Parts);
    serializer.CloseFile();

    serializer.OpenFile("testOut.json", true);
    serializer.GoToPath("main");
    Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_Unknown;
    serializer.ReadRealVector("aRealVec", rVec, offsetV);
    serializer.ReadRowsizeVector("aRSVec", rsVec0, offsetV);
    serializer.ReadSharedIndexVector("aIndexVec_shared", p_iVec0, offsetV);
    serializer.ReadSharedIndexVector("aIndexVec_shared_1", p_iVec1, offsetV);
    DNDS::index sizeRead;
    serializer.ReadUint8Array("aRealVecWithoutType", nullptr, sizeRead, offsetV);
    DNDS_assert(sizeRead == rVec2.size() * sizeof(real));
    serializer.ReadUint8Array("aRealVecWithoutType", (uint8_t *)rVec2.data(), sizeRead, offsetV);
    serializer.CloseFile();

    return 0;
}

// int main()
// {
//     using base64 = cppcodec::base64_rfc4648;

//     std::vector<uint8_t> decoded = base64::decode("AAAAAAAA8D8AAAAAAAAAQGZmZmZmZgpAAAAAAAAAEECamZmZmZkVQA==");
//     return 0;
// }