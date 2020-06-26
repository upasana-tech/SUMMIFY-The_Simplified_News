function saveFile() {
    	var data = document.getElementById("p1").textContent;
        // Convert the text to a BLOB.
        var textToBLOB = new Blob([data], { type: 'text/plain' });
        var sFileName = 'formData.txt';	   // The file to save the data.

        var newLink = document.createElement("a");
        newLink.download = sFileName;

        if (window.webkitURL != null) {
            newLink.href = window.webkitURL.createObjectURL(textToBLOB);
        }
        else {
            newLink.href = window.URL.createObjectURL(textToBLOB);
            newLink.style.display = "none";
            document.body.appendChild(newLink);
        }

        newLink.click();
        alert("File Successfully Saved.");
    }