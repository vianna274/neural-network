class PrintToFile:

  @staticmethod
  def print_accuracy(accuracy_list):
    cross_validation_iteration = 0
    file = open("acc.txt", "a")
    file.write("iteration_index acc\n")

    for accuracy in accuracy_list:
      file.write(str(cross_validation_iteration) + " " + str(accuracy) + "\n")
      cross_validation_iteration += 1

    file.close()

  @staticmethod
  def print_j(j_lists):
    cross_validation_iteration = 0
    file = open("j.txt", "a")
    file.write("iteration_index j_array\n")

    for j_list in j_lists:
      file.write(str(cross_validation_iteration) + " " + str(j_list) + "\n")
      cross_validation_iteration += 1

    file.close()

  @staticmethod
  def print_statistics(result):
    file = open("result.txt", "a")
    file.write(str(result))
    file.close()