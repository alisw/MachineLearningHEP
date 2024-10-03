#  © Copyright CERN 2018. All rights not expressly granted are reserved.  #
#                 Author: Gian.Michele.Innocenti@cern.ch                  #
# This program is free software: you can redistribute it and/or modify it #
#  under the terms of the GNU General Public License as published by the  #
# Free Software Foundation, either version 3 of the License, or (at your  #
# option) any later version. This program is distributed in the hope that #
#  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  #
#     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    #
#           See the GNU General Public License for more details.          #
#    You should have received a copy of the GNU General Public License    #
#   along with this program. if not, see <https://www.gnu.org/licenses/>. #

"""
Generate BDT cut variations
Author: Vit Kucera <vit.kucera@cern.ch>
"""


def main():
    n_steps = 5
    print_default = False

    dic_cuts = {
        "d0" : {
            "string": "mlBkgScore < %g",
            "cuts_default" : [0.02, 0.02, 0.02, 0.05, 0.06, 0.08, 0.08, 0.10, 0.10, 0.20, 0.25, 0.30],  # default
            "cuts_min" : [0.008, 0.008, 0.0087, 0.017, 0.024, 0.031, 0.028, 0.042, 0.038, 0.052, 0.067, 0.060],  # tight
            "cuts_max" : [0.045, 0.053, 0.054, 0.19, 0.22, 0.33, 0.46, 0.38, 0.50, 0.50, 0.50, 0.50]  # loose
        },
        "lc": {
            "string" : "mlPromptScore > %g",
            "cuts_default" : [0.97, 0.9, 0.9, 0.85, 0.85, 0.8, 0.8, 0.6, 0.6],  # default
            "cuts_min" : [0.961, 0.83, 0.84, 0.74, 0.74, 0.62, 0.63, 0.15, 0.15],  # loose
            "cuts_max" : [0.978, 0.94, 0.937, 0.915, 0.91, 0.89, 0.88, 0.85, 0.85]  # tight
        }
    }

    def format_list(str_format: str, values: list):
        return [str_format % val for val in values]

    def format_comment(comment: str):
        return f" # {comment}"

    for hf, cuts in dic_cuts.items():
        cuts_default = cuts["cuts_default"]
        fmt = cuts["string"]
        greater_than = ">" in fmt

        # Calculate steps
        step_down = [(minimum - default) / n_steps for minimum, default in zip(cuts["cuts_min"], cuts_default)]
        step_up = [(maximum - default) / n_steps for maximum, default in zip(cuts["cuts_max"], cuts_default)]
        list_down = []
        list_up = []

        # Calculate variations
        for i in range(n_steps):
            list_down.append([round(default + (i + 1) * step, 6) for default, step in zip(cuts_default, step_down)])
            list_up.append([round(default + (i + 1) * step, 6) for default, step in zip(cuts_default, step_up)])

        labels_down = [("loose" if greater_than else "tight") + f" {i + 1}" for i in range(n_steps)]
        labels_up = [("tight" if greater_than else "loose") + f" {i + 1}" for i in range(n_steps)]

        labels = list(reversed(labels_down))
        if print_default:
            labels += ["default"]
        labels += labels_up

        # Print flags and labels
        n_items = 2 * n_steps + int(print_default)
        prefix_item = "    - "

        print(f"{hf}:")
        act = f"{n_items * 'yes, '}"
        print(f"  activate: [{act[:-2]}]")
        print("  label:", labels)
        print("  use_cuts:", n_items * [True])

        # Print numeric variations
        print("  cuts_num:")
        for var, label in zip(reversed(list_down), reversed(labels_down)):
            print(prefix_item, var, format_comment(label))
        if print_default:
            print(prefix_item, cuts_default, format_comment("default"))
        for var, label in zip(list_up, labels_up):
            print(prefix_item, var, format_comment(label))

        # Print formatted variations
        print("  cuts:")
        for var, label in zip(reversed(list_down), reversed(labels_down)):
            print(prefix_item, format_list(fmt, var), format_comment(label))
        if print_default:
            print(prefix_item, format_list(fmt, cuts_default), format_comment("default"))
        for var, label in zip(list_up, labels_up):
            print(prefix_item, format_list(fmt, var), format_comment(label))


if __name__ == "__main__":
    main()
