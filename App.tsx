import React from "react";
import { StyleSheet, View } from "react-native";
import { SafeAreaProvider } from "react-native-safe-area-context";
import EvaluationScreen from "./src/screens/evaluationScreen";

export default function App() {
  return (
    <SafeAreaProvider>
      <View style={styles.container}>
        <EvaluationScreen />
      </View>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
});
