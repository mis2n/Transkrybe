const childProcess = require('child_process');

const pythonScript = 'script.py';
const environmentName = 'testenv';

//const command = `echo hello der!`
const command =`conda run -n ${environmentName} python ${pythonScript}`

const pythonProcess = childProcess.spawn(command, { shell: true });

pythonProcess.stdin.on('data', (data) => console.log(data.toString()));
pythonProcess.stderr.on('data', (data) => console.error(data.toString()));

pythonProcess.on('close', (code) => {
  console.log('Process Exited:', code);
});