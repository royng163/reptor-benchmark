import { Platform } from "react-native";

const log = (message: string) => {
  if (Platform.OS === "web") {
    console.log(message);
  } else {
    // For mobile platforms, consider using a logging library or native logging
    // Here we can use console.log for simplicity
    console.log(message);
  }
};

const error = (message: string) => {
  if (Platform.OS === "web") {
    console.error(message);
  } else {
    // For mobile platforms, consider using a logging library or native logging
    console.error(message);
  }
};

const info = (message: string) => {
  if (Platform.OS === "web") {
    console.info(message);
  } else {
    console.info(message);
  }
};

export default {
  log,
  error,
  info,
};
