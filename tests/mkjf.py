import json 
import glob
import os

from matplotlib.font_manager import json_dump


def make_directory(filename):
    try:
        os.mkdir(filename)
    except:
        print(filename,"exist!")
    return


JSON_BASE_FILENAME = "./dt/json/"
master = glob.glob(JSON_BASE_FILENAME+"70_10/*")
master.sort()

PEAK_STR_RANGE = slice(-7,-5)
TIMES_DATA_RANGE_4_JSON = slice(-10,-8)
TIMES_DATA_RANGE_4_FILE = slice(-15,-13)
ANGLE_DATA_RANGE_4_JSON = slice(-12,-10)
ANGLE_DATA_RANGE_4_FILE = slice(-18,-16)


ANGLE_LIST = [
    "00","05",
    "10","15",
    "20","25",
    "30","35",
    "40","45",
    "50","55",
    "60","65",
    "70","75",
    "80","85",
    "90","95",
]

TIMES_LIST = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
]
# # TIMES_LIST = [
# #     "01",
# # ]



for fin in master:
    peak_str = fin[PEAK_STR_RANGE]
    fin_angle = int(fin[ANGLE_DATA_RANGE_4_JSON])
    fin_times = int(fin[TIMES_DATA_RANGE_4_JSON])

    with open(fin) as f:
        jf_dict = json.load(f)
    f.close()

    for fout_times in TIMES_LIST:
        print("############")
        print("fin_json",fin)

        fout_angle_str = str(fin_angle).zfill(2)
        fout_times_str = str(fout_times).zfill(2)

        fout_dir_name = "{jb}{fta}_{fts}".format(
            jb = JSON_BASE_FILENAME,
            fta = fout_angle_str,
            fts = fout_times_str
        )

        fout_json_name = "{jb}{fas}_{fts}/{fas}{fts}_{peak}.json".format(
            jb = JSON_BASE_FILENAME,
            fas = fout_angle_str,
            fts = fout_times_str,
            peak = peak_str
        )

        print("fout_dir",fout_dir_name)
        print("fout_json",fout_json_name)

        jf_dict["dpar"]["dt_name"] = f"c{fout_angle_str}000{fout_times_str}"
        print("dataname",jf_dict["dpar"]["dt_name"])
        
        make_directory(fout_dir_name)
        json_dump(jf_dict,fout_json_name)



    del jf_dict






