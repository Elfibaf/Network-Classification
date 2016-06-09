#!/bin/bash

PREFIX=${1:-/trace}
TEST=./test-convert2
export LD_LIBRARY_PATH=../lib/.libs:../libpacketdump/.libs
#export LIBTRACEIO=directread

# Auckland I

# Auckland II
echo Testing Auckland II
for i in ${PREFIX}/auckland/2/*.gz; do
	echo -n " * $i: "
	${TEST} legacyatm:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

# Auckland IV
echo Testing Auckland IV
for i in ${PREFIX}/auckland/4/*.gz; do
	echo -n " * $i: "
	${TEST} legacyatm:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

# Auckland VI
echo Testing Auckland VI
for i in ${PREFIX}/auckland/6/*-[01].gz; do
	echo -n " * $i: "
	${TEST} legacyatm:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done
for i in ${PREFIX}/auckland/6/*-e[01].gz; do
	echo -n " * $i: "
	${TEST} legacyeth:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

# Auckland VII
echo Testing Auckland VIII
for i in ${PREFIX}/auckland/8/*.gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

# Auckland IX
echo Testing Auckland IX
for i in ${PREFIX}/auckland/9/*.gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

# IPLS I
echo Testing IPLS I
for i in ${PREFIX}/pma/long/ipls/1/*.gz; do
	echo -n " * $i: "
	${TEST} legacypos:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

# IPLS II
echo Testing IPLS II
for i in ${PREFIX}/pma/long/ipls/2/*.gz; do
	echo -n " * $i: "
	${TEST} legacypos:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

# IPLS III
echo Testing IPLS III
for i in ${PREFIX}/pma/long/ipls/3/*.gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

# Leipzig I
echo Testing Leipzig I 
for i in ${PREFIX}/pma/long/leip/1/*.gz; do
	echo -n " * $i: "
	${TEST} legacypos:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

# Leipzig II
echo Testing Leipzig II 
for i in ${PREFIX}/pma/long/leip/2/*-[01].gz; do
	echo -n " * $i: "
	${TEST} legacypos:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done
for i in ${PREFIX}/pma/long/leip/2/*-e.gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

# NZIX II
echo Testing NZIX II 
for i in ${PREFIX}/pma/long/nzix/2/*.gz; do
	echo -n " * $i: "
	${TEST} legacyeth:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

# SDSC I
echo Testing SDSC I
for i in ${PREFIX}/pma/long/sdag/1/*.gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done


# Waikato I
echo Testing Waikato I
for i in ${PREFIX}/waikato/1/*[0-9].gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

echo Testing Waikato II
for i in ${PREFIX}/waikato/2/*[0-9].gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

echo Testing Waikato III
for i in ${PREFIX}/waikato/3/*[0-9].gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

echo Testing Waikato IV
for i in ${PREFIX}/waikato/4/*[0-9].gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

echo Testing Waikato V
for i in ${PREFIX}/waikato/5/*[0-9].gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

echo Testing Waikato VI
for i in ${PREFIX}/waikato/6/*[0-9].gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done

echo Testing Waikato VIII
for i in ${PREFIX}/waikato/8/*[0-9].gz; do
	echo -n " * $i: "
	${TEST} erf:$i pcapfile:/scratch/salcock/libtrace_test/convert_test2.pcap && echo PASS
	#break
done




echo Testing complete
