def calc_metrics(fp_model, test_images, test_labels):
    # Evaluate the model on the test set
    fp_test_loss, fp_test_acc = fp_model.evaluate(test_images, test_labels, verbose=2)
    print("Test accuracy:", fp_test_acc)
    print("Test loss:", fp_test_loss)
